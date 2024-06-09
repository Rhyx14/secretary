from typing import Any
from ..data_recorder import Avg
class Record_Loss:
    def __init__(self,secretary,name_prefix='') -> None:
        self._secretary=secretary
        self._name_prefix=name_prefix
        pass

    def __call__(self, batch_len,batch_id,loss_value,epochs) -> Any:
        _key=Avg(f'{self._name_prefix}_training_loss_{epochs}',step=batch_id)
        self._secretary.data[_key]=loss_value
        self._secretary.print_solo(f'Epoch {epochs}: {batch_id}/{batch_len}, training loss: {loss_value:.5f} - avg: {self._secretary.data[_key]:.10f}',end='\r')
