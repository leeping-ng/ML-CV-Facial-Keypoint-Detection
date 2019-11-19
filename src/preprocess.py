import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torchvision import transforms

def prepare_data(path_train_csv, path_test_csv, path_train_images, path_test_images):
    """
    Takes in file paths and returns transformed datasets
    """
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    train_dataset = FacialKeypointsDataset(path_train_csv, path_train_images, transform=data_transform)
    test_dataset = FacialKeypointsDataset(path_test_csv, path_test_images, transform=data_transform)

    print("Data preprocessed and transformed.")
    print("No. of training data: ", len(train_dataset))
    print("No. of test data: ", len(test_dataset))

    return train_dataset, test_dataset


# Adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class FacialKeypointsDataset():
    """
    To return a dictionary of:
    {"image": image, "keypoints": keypoints}
    """
    def __init__(self, csv_file, path_base, transform=None):
        self.keypoints_frame = pd.read_csv(csv_file)
        self.path_base = path_base
        self.transform = transform
        
    def __len__(self):
        return len(self.keypoints_frame)
    
    # read images here instead of init to save memory
    def __getitem__(self, idx):

        image_name = self.keypoints_frame.iloc[idx, 0]
        image = mpimg.imread(os.path.join(self.path_base, image_name))
        
        # if image has an alpha color channel (4th), get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
            
        keypoints = self.keypoints_frame.iloc[idx, 1:]
        # change from pandas series to numpy array, reshape to (num_keypoints, 2)
        keypoints = keypoints.to_numpy().astype('float').reshape(-1, 2)
        
        sample = {"image": image, "keypoints": keypoints}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class Normalize(object):
    """
    To convert a color image to grayscale values with a range of [0,1] 
    and normalize the keypoints to be in a range of about [-1, 1]
    """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        keypoints_copy = np.copy(keypoints)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        keypoints_copy = (keypoints_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': keypoints_copy}


class Rescale(object):
    """
    To rescale an image to a desired size
    """
    def __init__(self, output_size):
        # If tuple, output is matched to output_size. 
        # If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        keypoints = keypoints * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': keypoints}


class RandomCrop(object):
    """
    To crop an image randomly
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            #If int, square crop is made
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        keypoints = keypoints - [left, top]

        return {'image': image, 'keypoints': keypoints}


class ToTensor(object):
    """
    To convert numpy images to torch images
    """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}