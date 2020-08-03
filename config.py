from collections import OrderedDict as edict
import torch

cfg = edict()
cfg.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
cfg.saveFolder = './predict/TrajGRUWeightedFocalAll3/'
cfg.checkpoint = '/home/tsingzao/projects/MetNet_TrajGRU/checkpoint_TrajGRU_Weighted_FocalAll/checkpoint_003.pth.tar'
cfg.filePath = "./MetNet/test.txt"
cfg.dataFolder = '/home/tsingzao/Dataset/test/'
# cfg.dataFolder = './testData/'


