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
    return Image.open(path).convert('RGB')


class VideoLoader(torch.utils.data.Dataset):

    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):
        self.dataset_object = JpegDataset(csv_file_input, csv_file_labels, root)

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

            item = self.csv_data[index]
            img_paths = self.get_frame_names(item.path)
            imgs = []

            for img_path in img_paths:
                try:
                    img = self.loader(img_path)
                    img = self.transform(img)
                    imgs.append(torch.unsqueeze(img, 0))
                except Exception as e:
                    print(f"Error loading image at path: {img_path}. {e}")
                    continue

            if len(imgs) == 0:
                raise RuntimeError("No images were loaded for this data point.")

            target_idx = self.classes_dict[item.label]
            data = torch.cat(imgs)
            data = data.permute(1, 0, 2, 3)
            return data, target_idx
        except Exception as ex:
            logging.warning(ex)

    def __len__(self):
        return len(self.csv_data)


    def get_frame_names(self, path):
        try:
            frame_names = []
            for ext in IMG_EXTENSIONS:
                frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
            frame_names = list(sorted(frame_names))
            num_frames = len(frame_names)
            if self.nclips > -1:
                num_frames_necessary = self.clip_size * self.nclips * self.step_size
            else:
                num_frames_necessary = num_frames
            offset = 0
            if num_frames_necessary > num_frames:
                if frame_names:
                    frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
                else:
                    print(f"No frames found in directory: {path}")
                    return []
            elif num_frames_necessary < num_frames:
                diff = (num_frames - num_frames_necessary)
                if not self.is_val:
                    offset = np.random.randint(0, diff)

            frame_names = frame_names[offset:num_frames_necessary + offset:self.step_size]
            return frame_names
        except Exception as ex:
            logging.warning(ex)



if __name__ == '__main__':
    transform = Compose([
                        CenterCrop(84),
                        ToTensor(),
                        ])
    loader = VideoLoader(root="/hdd/20bn-datasets/20bn-jester-v1/",
                         csv_file_input="csv_files/jester-v1-validation.csv",
                         csv_file_labels="csv_files/jester-v1-labels.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=False,
                         transform=transform,
                         loader=default_loader)


    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=5, pin_memory=True)