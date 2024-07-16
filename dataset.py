import pandas as pd
import random as rd
import torch
import tifffile 
import numpy as np

from torch.utils.data import Dataset


##########
# Function that reads .pq file and extract file paths for stack, rfv, and ground truth
##########
def read_pq_file(file_path, one_sample = False):
    '''Make sure that they're in a list format for it to work with data loader class'''
    df = pd.read_parquet(file_path)

    if one_sample:
        random_num = 456 # rd.randrange(500)
        
        stack_path = [df.iloc[random_num]['stack_scat_path']] # 9 channel
        rfv_path = [df.iloc[random_num]['rfv_scat_path']] # 24 channel
        ground_truth_path = [df.iloc[random_num]['gt_path']]

        print("stack_path: ", stack_path)
        print("rfv_path: ", rfv_path)
        print("ground_truth path: ", ground_truth_path, "\n")
    else:
        stack_path = df['stack_scat_path'].tolist()
        rfv_path = df['rfv_scat_path'].tolist()
        ground_truth_path = df['gt_path'].tolist()

    return stack_path, rfv_path, ground_truth_path


##########
# Creating class for dataset without patches
##########
class NoPatchDataset(Dataset):
    def __init__(self, stack_paths, rfv_paths, truth_paths):
        super(NoPatchDataset, self).__init__()
        self.stack_paths = stack_paths
        self.rfv_paths = rfv_paths
        self.truth_paths = truth_paths

    def __len__(self):
        '''Return length of the dataset (# of patces per sample)'''
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

        '''Converting to tensor'''
        stack_tensor = torch.tensor(stack, dtype=torch.float32)
        rfv_tensor = torch.tensor(rfv, dtype=torch.float32)
        truth_tensor = torch.tensor(truth, dtype=torch.float32)

        # print(f"Shapes of NoPatchDataset | stack: {stack_tensor.shape}, rfv: {rfv_tensor.shape}, truth: {truth_tensor.shape}")

        return stack_tensor, rfv_tensor, truth_tensor


##########
# Creating class for single patch dataset
# Generating one random patch with consistent location and size across all images for 
# stack, rfv, and truth
##########
class SinglePatchDataset(Dataset):
    def __init__(self, stack_paths, rfv_paths, truth_paths, patch_size):
        super(SinglePatchDataset, self).__init__()
        self.stack_paths = stack_paths
        self.rfv_paths = rfv_paths
        self.truth_paths = truth_paths
        self.patch_size = patch_size

    def __len__(self):
        '''Return number of patches per sample'''
        return len(self.stack_paths)
    
    def __getitem__(self, idx): 
        '''Reading .tiff files'''
        stack = tifffile.imread(self.stack_paths[idx])
        rfv = tifffile.imread(self.rfv_paths[idx])
        truth = tifffile.imread(self.truth_paths[idx])

        '''Calculating random starting points for the patch'''
        H, W = stack.shape[-2], stack.shape[-1]

        row_start = torch.randint(0, H - self.patch_size + 1, (1,)).item()
        col_start = torch.randint(0, W - self.patch_size + 1, (1,)).item()

        '''Extracting patches'''
        stack_patch = stack[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
        rfv_patch = rfv[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
        truth_patch = truth[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]

        '''Normalizing'''
        stack_patch = stack_patch.astype('float32') / 65535.0
        rfv_patch = rfv_patch.astype('float32') / 65535.0
        truth_patch = truth_patch.astype('float32') / 65535.0

        '''Converting to tensor'''
        stack_patch_tensor = torch.tensor(stack_patch, dtype=torch.float32)
        rfv_patch_tensor = torch.tensor(rfv_patch, dtype=torch.float32)
        truth_patch_tensor = torch.tensor(truth_patch, dtype=torch.float32)

        # print(f"Shapes of PatchDataset | stack: {stack_patch_tensor.shape}, rfv: {rfv_patch_tensor.shape}, truth: {truth_patch_tensor.shape}")

        return stack_patch_tensor, rfv_patch_tensor, truth_patch_tensor


##########
# Creating class for patch dataset
##########
class PatchDataset(Dataset):
    def __init__(self, stack_paths, rfv_paths, truth_paths, patch_size):
        super(PatchDataset, self).__init__()
        self.stack_paths = stack_paths
        self.rfv_paths = rfv_paths
        self.truth_paths = truth_paths
        self.patch_size = patch_size

    def __len__(self):
        '''Return number of patches per sample'''
        image = tifffile.imread(self.stack_paths[0])
        H, W = image.shape[-2], image.shape[-1]
    
        total_patches = (H // self.patch_size) * (W // self.patch_size) * len(self.stack_paths)
        # print("Total patches: ", total_patches)
        return total_patches
    
    def __getitem__(self, idx):
        '''Retrieve a specific patch
        Find which image the patch belongs to'''
        accumulated_patches = 0
        for i, path in enumerate(self.stack_paths):
            image = tifffile.imread(path)
            h, w = image.shape[-2], image.shape[-1]
            num_patches_h = h // self.patch_size
            num_patches_w = w // self.patch_size
            num_patches = num_patches_h * num_patches_w

            if accumulated_patches + num_patches > idx:
                image_idx = i
                patch_idx = idx - accumulated_patches
                break
            accumulated_patches += num_patches

        '''Reading .tiff files'''
        stack = tifffile.imread(self.stack_paths[image_idx])
        rfv = tifffile.imread(self.rfv_paths[image_idx])
        truth = tifffile.imread(self.truth_paths[image_idx])

        '''Calculating row and column for the patch'''
        patch_row = patch_idx // num_patches_w
        patch_col = patch_idx % num_patches_w

        row_start = patch_row * self.patch_size
        col_start = patch_col * self.patch_size

        '''Extracting patches'''
        stack_patch = stack[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
        rfv_patch = rfv[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]
        truth_patch = truth[:, row_start:row_start + self.patch_size, col_start:col_start + self.patch_size]

        '''Normalizing'''
        stack_patch = stack_patch.astype('float32') / 65535.0
        rfv_patch = rfv_patch.astype('float32') / 65535.0
        truth_patch = truth_patch.astype('float32') / 65535.0

        '''Converting to tensor'''
        stack_patch_tensor = torch.tensor(stack_patch, dtype=torch.float32)
        rfv_patch_tensor = torch.tensor(rfv_patch, dtype=torch.float32)
        truth_patch_tensor = torch.tensor(truth_patch, dtype=torch.float32)

        # print(f"Shapes of PatchDataset | stack: {stack_patch_tensor.shape}, rfv: {rfv_patch_tensor.shape}, truth: {truth_patch_tensor.shape}")

        return stack_patch_tensor, rfv_patch_tensor, truth_patch_tensor