import pandas as pd
import random as rd
import torch
import tifffile 

from torch.utils.data import Dataset


def read_pq_file(file_path):
    random_num = 5 # rd.randrange(500)

    df = pd.read_parquet(file_path)
    
    stack_path = df.iloc[random_num]['stack_scat_path'] # 9 channel
    rfv_path = df.iloc[random_num]['rfv_scat_path'] # 24 channel
    ground_truth_path = df.iloc[random_num]['gt_path']

    return stack_path, rfv_path, ground_truth_path
    

##########
# Creating Data Loader
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
    
    def __getitem__(self, idx):
        stack = tifffile.imread(self.stack_paths[idx])
        rfv = tifffile.imread(self.rfv_paths[idx])
        truth = tifffile.imread(self.truth_paths[idx])

        '''Converting tiff into tensor and normalizing by dividing values by 65535.0
        because the tiff files are saved in uint16, which indicates max value is 65535.0
        (each pixel value ranges from 0 to 65535  for uint16)
        Note: Look at data_visualization.ipynb for more details'''
        stack = torch.tensor(stack, dtype=torch.float32) / 65535.0
        rfv = torch.tensor(rfv, dtype=torch.float32) / 65535.0
        truth = torch.tensor(truth, dtype=torch.float32) / 65535.0

        return stack, rfv, truth