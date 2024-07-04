'''
Pre-defined enum classes
'''
from enum import Enum
class CLASSIFICATION(Enum):
    ImageNet1K=0
    CIFAR10=1
    CIFAR100=2
    tiny_ImageNet=3
    CIFAR10_DVS=4
    MNIST=5
    F_MNIST=6
    
    DVS_GESTURE=5
    N_MNIST=7
    SHD=6

class SEMANTIC_SEGMENTATION(Enum):
    SBD=0
    COCO2012=1
    CamVid=2
    Supervisely_Person=3    