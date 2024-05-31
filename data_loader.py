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

    stack_paths = df['stack_scat_path'].tolist() # 9 channel
    rfv_paths = df['rfv_scat_path'].tolist() # 24 channel
    ground_truth_paths = df['gt_path'].tolist()

    return stack_paths, rfv_paths, ground_truth_paths


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
    
    def __getitem__(self, index):
        stack = tifffile.imread(self.stack_paths[index])
        rfv = tifffile.imread(self.rfv_paths[index])
        truth = tifffile.imread(self.truth_paths[index])

        # Converting tiff into tensor
        stack = torch.tensor(stack, dtype=torch.float32)
        rfv = torch.tensor(rfv, dtype=torch.float32)
        truth = torch.tensor(truth, dtype=torch.float32)

        return stack, rfv, truth