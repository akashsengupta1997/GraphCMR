import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import config

class SportsVideosEvalDataset(Dataset):
    def __init__(self, dataset_path, img_wh):
        super(SportsVideosEvalDataset, self).__init__()

        # Data
        data = np.load(os.path.join(dataset_path, 'sports_videos_eval.npz'))

        self.frame_paths = data['frames_paths']
        self.vertices = data['vertices']
        self.body_shapes = data['shapes']
        self.genders = data['genders']

        assert len(self.frame_paths) == len(self.vertices) == len(self.body_shapes) == len(self.genders)

        self.img_wh = img_wh
        self.normalize_img = Normalize(mean=config.IMG_NORM_MEAN, std=config.IMG_NORM_STD)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        frame_path = self.frame_paths[index]
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])/255.0

        # Targets
        vertices = self.vertices[index]
        shape = self.body_shapes[index]
        gender = self.genders[index]

        img = torch.from_numpy(img).float()
        vertices = torch.from_numpy(vertices).float()
        shape = torch.from_numpy(shape).float()

        # Process image
        input = self.normalize_img(img)

        return {'input': input,
                'vis_img': img,
                'shape': shape,
                'vertices': vertices,
                'frame_path': frame_path,
                'gender': gender}