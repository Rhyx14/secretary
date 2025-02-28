from torch.autograd.graph import saved_tensors_hooks
import torch
from typing import Any, Tuple,Deque,Optional
from types import MethodType
import collections
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper

class _FreeEventQueue:
    """
    A copy of torch.distributed.fsdp._limiter_utils._FreeEventQueue, where torch version >= 2.0.

    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[torch.cuda.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: torch.cuda.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Optional[torch.cuda.Event]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None

class faster_save_on_cpu(saved_tensors_hooks):
    '''
    A faster version of torch.autograd.graph.save_on_cpu

    using dedicated stream to speed up tensor transfering of H2D / D2H

    https://github.com/pytorch/pytorch/issues/106360
    '''
    
    copy = None
    queue = _FreeEventQueue()

    def __init__(self):
        if faster_save_on_cpu.copy is None:
            # make sure we only create the h2d/d2h stream once
            faster_save_on_cpu.copy = torch.cuda.Stream()

        def pack_to_cpu(tensor: torch.Tensor):
            # Here, we may redundantly save full weights tensors to cpu. 
            # For example, by default, a linear layer will save its input, weight 
            # and bias for backward. However, if we already have FSDP, we probably 
            # don't want save the **unsharded** weight and bias to CPU and want to 
            # leave them on GPU to reduce memory transfers. Is there a way to skip 
            # saving weight tensors?

            assert not tensor.is_sparse  # not yet supported
            if not tensor.is_cuda:
                return None, tensor

            self.limit_prefetch()
            faster_save_on_cpu.copy.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(faster_save_on_cpu.copy):
                cpu_tensor = tensor.to("cpu", non_blocking=True)
            self.record_event()
            tensor.record_stream(faster_save_on_cpu.copy)
            return (tensor.device, cpu_tensor)

        def unpack_from_cpu(packed: Tuple[torch.device, torch.Tensor]):
            device, cpu_tensor = packed
            if device is None:
                return cpu_tensor

            self.limit_prefetch()
            with torch.cuda.stream(faster_save_on_cpu.copy):
                # here, we must allocate tensor on the copy stream to allow
                # copy to start early. If pytorch can provide a method to
                # allocate an empty block **immediately** rather than reusing a
                # previous block on the compute stream, the problem below can be solved. 
                tensor = cpu_tensor.to(device, non_blocking=True)
            self.record_event()

            # subsequent gradient computation need to wait for h2d to complete
            torch.cuda.current_stream().wait_stream(faster_save_on_cpu.copy)

            # Now we have a problem. Ideally, we want to avoid this memcpyD2D to save time and memory, 
            # but the tensor is allocated on a different stream as opposed to the compute stream, so
            # to make this work, we need to call .record_stream after this tensor's lifetime ends
            # during backward. However, autogard seems like a blackbox to me, making it not 
            # practical to insert a .record_stream(). As a workaround, we can clone this tensor. 

            # I've tried hacking the destructor of out_tensor by subclassing and overriding Tensor's __del__,
            # but this introduced significant python overhead due to all subsequent tensors having to call
            # .record_stream(). 
            out_tensor = tensor.clone()
            tensor.record_stream(torch.cuda.current_stream())
            return out_tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)

    def limit_prefetch(self):
        # limit the number of h2d/d2h submitted to reduce peak memory usage, like FSDP's limit_all_gathers option
        prev_ev = self.queue.dequeue_if_needed()
        if prev_ev:
            prev_ev.synchronize()

    def record_event(self):
        event = torch.cuda.Event()
        event.record(faster_save_on_cpu.copy)
        self.queue.enqueue(event)

class _Offload_Forward():
    def __init__(self,original) -> None:
        self.original=original
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with faster_save_on_cpu():
            rslt=self.original(*args,**kwds)
        return rslt

def offload_module(module_type,net:torch.nn.Module,ratio=1.):
    '''
    Offload the interim tensor to cpu for reducing VRAM cost 
    '''
    _module_count=0.
    for module in net.modules():
        if isinstance(module,module_type):
            _module_count+=1

    _changed=0.
    for module in net.modules():
        if isinstance(module,module_type):
            _changed+=1
            if _changed / _module_count <= ratio:
                if not isinstance(module,torch.nn.Module): 
                    raise TypeError(f'Module: {module} should also be a subclass of torch.nn.Module.')
                module.forward=_Offload_Forward(module.forward)

def restore_offload(module_type,net:torch.nn.Module):
    for module in net.modules():
        if isinstance(module,module_type):
            if not isinstance(module,torch.nn.Module): raise TypeError(f'Module: {module} should also be a subclass of torch.nn.Module.')
            module.forward=MethodType(module_type.forward,module)