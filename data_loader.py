import pandas as pd
import random as rd
import torch
import tifffile 

from torch.utils.data import Dataset

"""
Function to:
1) retrieve information from .pq file
2) Randomly selects one .pq file, 
3) Read its corresponding .tiff file 
4) Converting .tiff to torch tensors
"""
def read_pq_file(file_path):
    random_num = rd.randrange(500)

    df = pd.read_parquet(file_path)
    
    stack_path = df.iloc[random_num]['stack_scat_path'] # 9 channel
    rfv_path = df.iloc[random_num]['rfv_scat_path'] # 24 channel
    ground_truth_path = df.iloc[random_num]['gt_path']

    return stack_path, rfv_path, ground_truth_path


##########
# Dataset class s.t. we can work with one or more tiff files
##########
class TiffDataset(Dataset):
    def __init__(self, stack_paths, rfv_paths, truth_paths):
        super(TiffDataset, self).__init__()
        self.stack_paths = stack_paths
        self.rfv_paths = rfv_paths
        self.truth_paths = truth_paths

    def __len__(self):
        '''Return length of the dataset.'''
        return len(self.stack_paths)
    
    def __getitem__(self): # index
        stack = tifffile.imread(self.stack_paths)
        rfv = tifffile.imread(self.rfv_paths)
        truth = tifffile.imread(self.truth_paths)

        '''Converting tiff into tensor and normalizing by dividing values by 255
        because the tiff files are saved in uint8, which indicates max value is 255
        (each pixel value ranges from 0 to 255  for uint8)'''
        stack = torch.tensor(stack, dtype=torch.float32) / 255.0
        rfv = torch.tensor(rfv, dtype=torch.float32) / 255.0
        truth = torch.tensor(truth, dtype=torch.float32) / 255.0

        return stack, rfv, truth