import os
import glob
import numpy as np
import torch
import logging
from PIL import Image
from data_parser import JpegDataset
from torchvision.transforms import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def default_loader(path):
    try:
        image = Image.open(path)
        image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error opening or converting image: {e}")
        return None

class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, csv_file, labels, clip_size,nclips, step_size, is_val, transform=None,loader=default_loader):
        self.dataset_object = JpegDataset(csv_file, labels, root)
        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.loader = loader
        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val


    def __getitem__(self, index):
        try:
            index = self.csv_data[index]
            paths = self.get_frames(index.path)
            images = []
            for path in paths:
                try:
                    image = self.loader(path)
                    image = self.transform(image)
                    images.append(torch.unsqueeze(image, 0))
                except Exception as ex:
                    logging.warning(f"__getitem__ method's exception {ex} getting in loading the image at path : {path}")
                    continue

            if len(images) == 0:
                raise RuntimeError("There are no images loaded for data at {index}")
            data = torch.cat(images)
            data = data.permute(1, 0, 2, 3)
            index = self.classes_dict[index.label]
            return data, index
        except Exception as ex:
            logging.warning(f' __getitem__ method is giving exception {0}',ex)

    def __len__(self):
        try: 
            return len(self.csv_data)
        except Exception as ex:
            logging.warning(f'get length method exception : {0}',ex)


    def get_frames(self, path):
        try:
            frames = []
            for ext in IMG_EXTENSIONS:
                frames.extend(glob.glob(os.path.join(path, "*" + ext)))
            frames = list(sorted(frames))
            num = len(frames)

            if self.nclips <= -1:
                frames_necessary = num
            else:
                frames_necessary = self.clip_size * self.nclips * self.step_size
                

            offset = 0
            if frames_necessary < num:
                if not self.is_val:
                    offset = np.random.randint(0, (num - frames_necessary))
            elif frames_necessary > num:
                if frames:
                    frames += [frames[-1]] * (frames_necessary - num)
                else:
                    print(f"No frames found in directory: {path}")
                    return []  

            frames = frames[offset:frames_necessary + offset:self.step_size]
            return frames
        except Exception as ex:
            logging.warning(f"get_frames method exception {ex}")


if __name__ == '__main__':
    transform = Compose([CenterCrop(84),ToTensor()])
    loader = VideoFolder(root="/hdd/20bn-datasets/20bn-jester-v1/",
                         csv_file="csv_files/jester-v1-validation.csv",
                         labels="csv_files/jester-v1-labels.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=False,
                         transform=transform,
                         loader=default_loader)
    train_loader = torch.utils.data.DataLoader(loader,batch_size=10, shuffle=False, num_workers=5, pin_memory=True)