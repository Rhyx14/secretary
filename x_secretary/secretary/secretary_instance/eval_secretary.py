from ..secretary_base import Secretary_base
from pathlib import Path
from loguru import logger
from ...utils.time import get_str_time
import sys
LOGGER_FMT='<blue>{time:YYYY-MM-DD HH:mm:ss Z}</blue> [<level>{level}</level>] <green>{name}:{line}</green><yellow>#</yellow> {message}'
class Eval_Secretary(Secretary_base):
    def __init__(self,name='Eval_Secretary',saved_dir='.',log=False,logging_level='INFO',log_file_suffix='') -> None:
        super().__init__(working_dir=Path(saved_dir))

        if log:
            logger.remove()
            logger.add(sys.stderr,format=LOGGER_FMT)
            # 保存日志
            self._log_file_handler=logger.add(str(self._working_dir/f'eval_{get_str_time()}{log_file_suffix}.txt'),level=logging_level,format=LOGGER_FMT)
        pass