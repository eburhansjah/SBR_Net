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
        print("ground_truth path: ", ground_truth_path, "\n")
    else:
        stack_path = df['stack_scat_path'].tolist()
        rfv_path = df['rfv_scat_path'].tolist()
        ground_truth_path = df['gt_path'].tolist()

    return stack_path, rfv_path, ground_truth_path
    

##########
# Creating Patchify class
# Reference: https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6
##########
class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        batch_size, channels, H, W = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(batch_size, channels, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B num_patches c p p )
        return a


##########
# Creating Data Loader
##########
class TiffDataset(Dataset):
    def __init__(self, stack_paths, rfv_paths, truth_paths, patch_size):
        super(TiffDataset, self).__init__()
        self.stack_paths = stack_paths
        self.rfv_paths = rfv_paths
        self.truth_paths = truth_paths

        self.patchify = Patchify(patch_size)

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

        stack_tensor = torch.tensor(stack, dtype=torch.float32)
        rfv_tensor = torch.tensor(rfv, dtype=torch.float32)
        truth_tensor = torch.tensor(truth, dtype=torch.float32)

        # Adding batch dimension [batch=1, channel, H, W]
        if len(stack_tensor.shape) == 3:
            stack_tensor = stack_tensor.unsqueeze(0)
        if len(rfv_tensor.shape) == 3:
            rfv_tensor = rfv_tensor.unsqueeze(0)
        if len(truth_tensor.shape) == 3:
            truth_tensor = truth_tensor.unsqueeze(0)

        # Apply Patchify on dataset
        stack_patches = self.patchify(stack_tensor)
        rfv_patches = self.patchify(rfv_tensor)
        truth_patches = self.patchify(truth_tensor)

        print("Normalized and patchified images")

        return stack_tensor, rfv_tensor, truth_tensor