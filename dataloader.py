import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=None, resize_width=None, repeat=1):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.repeat = repeat
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path)
        img = self.data_preproccess(img)
        label = np.array(label)
        return img, label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(';')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list

    def load_data(self, path):
        image = cv2.imread(path)
        return image

    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data
