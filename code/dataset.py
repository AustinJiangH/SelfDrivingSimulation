import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch

def read_driving_records(data_directory, test_fraction=0.1, valid_fraction=0.1):
    """Read data from file and split data into train, valid, test subsets
    
    Parameters
    ----------
    data_directory : str
        directory of the dataset used, which should contain a subdirectory 'IMG/' and a file 'driving_log.csv' 
    test_fraction : float
    valid_fraction : float

    Returns
    -------
    train_records : pd.DataFrame
    valid_records : pd.DataFrame
    test_records : pd.DataFrame
    """

    # read data from csv file
    data_directory = os.path.join(data_directory, 'driving_log.csv').replace('\\','/')
    data_col = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(data_directory, names= data_col)


    # train test valid split 
    data_len = data.shape[0]
    train_len = int(data_len * (1 - valid_fraction - test_fraction))
    valid_len = int(data_len * valid_fraction)
    print(f'Length of train: {train_len}, valid: {valid_len}, test: {data_len - train_len - valid_len}')
    train_records, valid_records, test_records  = np.split(data.sample(frac=1, random_state=42), [train_len, train_len + valid_len])

    return train_records, valid_records, test_records


def read_driving_image(data_directory, image_name):
    """Read an image to torch tensor
    
    Parameters
    ----------
    data_directory : str
        directory of the dataset used, which should contain a subdirectory 'IMG/' and a file 'driving_log.csv' 
    image_name : str
        file name of the image

    Returns
    -------
    current_image : torch.tensor
    """
    image_path = os.path.join(data_directory, 'IMG', image_name.split('\\')[-1]).replace('\\','/')
    current_image = read_image(image_path)
    current_image = current_image[:,65:-25,:]
    return current_image.float()


class DrivingDataset(Dataset):

    def __init__(self, data_directory, records, transform=None):
        self.records = records
        self.data_directory = data_directory
        self.transform = transform

    def __getitem__(self, index):
        batch_records = self.records.iloc[index]
        steering_angle = float(batch_records[3])


        # read images 
        center_image = read_driving_image(self.data_directory, batch_records[0])
        left_image = read_driving_image(self.data_directory, batch_records[1])
        right_image = read_driving_image(self.data_directory, batch_records[2])

        # read steering angles
        center_steering_angle = steering_angle
        left_steering_angle = steering_angle + 0.5
        right_steering_angle = steering_angle - 0.5

        # transform images if needed
        if self.transform is not None:
            center_image = self.transform(center_image)
            left_image   = self.transform(left_image)
            right_image  = self.transform(right_image)

        return (center_image, center_steering_angle), (left_image, left_steering_angle), (right_image, right_steering_angle)

    def __len__(self):
        return len(self.records)



def get_driving_data_loaders(batch_size, train_dataset, valid_dataset, test_dataset,  num_workers=0):

    valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader

