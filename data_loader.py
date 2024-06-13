import pandas as pd
import random as rd
import torch
import tifffile 

from torch.utils.data import Dataset


def read_pq_file(file_path, one_sample = False):
    # Make sure that they're in a list format for it to work with data loader class
    df = pd.read_parquet(file_path)

    if one_sample:
        random_num = 456 # rd.randrange(500)
        
        stack_path = [df.iloc[random_num]['stack_scat_path']] # 9 channel
        rfv_path = [df.iloc[random_num]['rfv_scat_path']] # 24 channel
        ground_truth_path = [df.iloc[random_num]['gt_path']]

        print("stack_path: ", stack_path)
        print("rfv_path: ", rfv_path)
        print("ground_truth path: ", ground_truth_path)
    else:
        stack_path = df['stack_scat_path'].tolist()
        rfv_path = df['rfv_scat_path'].tolist()
        ground_truth_path = df['gt_path'].tolist()

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
        '''Reading .tiff files'''
        stack = tifffile.imread(self.stack_paths[idx])
        rfv = tifffile.imread(self.rfv_paths[idx])
        truth = tifffile.imread(self.truth_paths[idx])

        '''Convert to float32 to work with torch.tensor before normalizing.

        Normalizing by dividing tensor values by 65535.0 because the 
        tiff files are saved in uint16, which indicates max value is 65535.0
        (each pixel value ranges from 0 to 65535  for uint16)

        Note: Look at data_visualization.ipynb for more details'''
        stack = stack.astype('float32') / 65535.0
        rfv = rfv.astype('float32') / 65535.0
        truth = truth.astype('float32') / 65535.0

        stack_tensor = torch.tensor(stack, dtype=torch.float32)
        rfv_tensor = torch.tensor(rfv, dtype=torch.float32)
        truth_tensor = torch.tensor(truth, dtype=torch.float32)

        print("Normalized images")

        return stack_tensor, rfv_tensor, truth_tensor