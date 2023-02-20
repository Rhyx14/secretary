from collections import defaultdict
import json
import os
class data_recorder:
    def __init__(self) -> None:
        self.serial_data=defaultdict(list)
        self.serial_data_index=defaultdict(list)
        self.data=defaultdict(float)
        pass

    def record_serial_data(self,key:str,index,value):
        self.serial_data[key].append(value)
        self.serial_data_index[key].append(index)
        pass

    def record_data(self,key:str,value):
        self.data[key]=value

    def save(self,path):
        with open(os.path.join(path,'record_data.json'),'w') as f:
            json.dump([self.serial_data,self.serial_data_index,self.data],f)