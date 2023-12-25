from collections import defaultdict
import json
import os
class Avg():
    def __init__(self,key:object,step=0) -> None:
        '''
        A wrapper for get / set moving average value
        '''
        self.key=key
        self.step=step
        pass

class Serial():
    def __init__(self,key,index=0) -> None:
        '''
        A wrapper for get / set serial value
        '''
        self.key=key
        self.index=index
        pass

class data_recorder:

    def __init__(self) -> None:
        self._serial_data=defaultdict(list)
        self._avg_data=defaultdict(float)
        self._normal_data=defaultdict(float)
        pass

    def save(self,path):
        with open(os.path.join(path,'record_data.json'),'w') as f:
            json.dump(
                {
                    "normal data":self._normal_data,
                    "serial data":self._serial_data,
                    "average data":self._avg_data,
                },
            f)

    def __getitem__(self,key):
        # 移动平均数
        if isinstance(key,Avg):
            return self._avg_data[key.key]
        # 序列数据
        elif isinstance(key,Serial):
            return self._serial_data[key.key][key.index]
        else:
            return self._normal_data[key]
    
    def __setitem__(self,key,value):
        # 移动平均数
        if isinstance(key,Avg):
            self._avg_data[key.key]=(self._avg_data[key.key]*key.step+ value)/(key.step+1)
        # 序列数据
        elif isinstance(key,Serial):
            self._serial_data[key.key].append(value)
        else:
            self._normal_data[key]=value
