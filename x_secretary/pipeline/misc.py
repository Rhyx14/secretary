from typing import Any
from ..data_recorder import Avg
class Record_Loss:
    '''
    Record loss
    ---
    Average value: f'{self._name_prefix}_training_loss_{epochs}'

    And print(solo): 

        f'Epoch {epochs}: {batch_id}/{batch_len}, training loss: {loss_value:.5f} - avg: {self._secretary.data[_key]:.10f}',end='\\r'
    '''
    def __init__(self,secretary,name_prefix='') -> None:
        self._secretary=secretary
        self._name_prefix=name_prefix
        pass

    def __call__(self, training_status) -> Any:
        ep=training_status['ep']
        b_id=training_status['batch_id']
        batch_len=training_status['batch_len']
        loss=training_status['loss']

        _key=Avg(f'{self._name_prefix}_training_loss_{ep}',step=b_id)
        
        self._secretary.data[_key]=loss
        self._secretary.print_solo(f'Epoch {ep}: {b_id}/{batch_len}, training loss: {loss:.5f} - avg: {self._secretary.data[_key]:.10f}',end='\r')

from tqdm import tqdm
import torch.distributed as dist
def DDP_progressbar(iterator):
    if dist.is_torchelastic_launched() and dist.get_rank()!=0:
        return iterator
    else:
        return tqdm(iterator,leave=False)