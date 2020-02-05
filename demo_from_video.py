#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import Mesh
from models import CMR
from utils.imutils import crop
from utils.renderer import Renderer
import config as cfg


def process_image(img_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    img = cv2.imread(img_file)[:, :,
          ::-1].copy()  # PyTorch does not support negative stride at the moment
    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def predict_on_frames(args):
    # Load model
    mesh = Mesh(device=device)
    # Our pretrained networks have 5 residual blocks with 256 channels.
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint, device=device)
    model.to(device)
    model.eval()

    image_paths = [os.path.join(args.in_folder, f) for f in sorted(os.listdir(args.in_folder))
                   if f.endswith('.png')]
    print('Predicting on all png images in {}'.format(args.in_folder))

    all_vertices = []
    all_vertices_smpl = []
    all_cams = []

    for image_path in image_paths:
        print("Image: ", image_path)
        # Preprocess input image and generate predictions
        img, norm_img = process_image(image_path, input_res=cfg.INPUT_RES)
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, pred_camera, _, _ = model(norm_img)

        pred_vertices = pred_vertices.cpu().numpy()
        pred_vertices_smpl = pred_vertices_smpl.cpu().numpy()
        pred_camera = pred_camera.cpu().numpy()

        all_vertices.append(pred_vertices)
        all_vertices_smpl.append(pred_vertices_smpl)
        all_cams.append(pred_camera)

    # Save predictions as pkl
    all_vertices = np.concatenate(all_vertices, axis=0)
    all_vertices_smpl = np.concatenate(all_vertices_smpl, axis=0)
    all_cams = np.concatenate(all_cams, axis=0)

    pred_dict = {'verts': all_vertices,
                 'verts_smpl': all_vertices_smpl,
                 'pred_cam': all_cams}
    if args.out_folder == 'dataset':
        out_folder = args.in_folder.replace('cropped_frames', 'cmr_results')
    else:
        out_folder = args.out_folder
    print('Saving to', os.path.join(out_folder, 'cmr_results.pkl'))
    os.makedirs(out_folder)
    for key in pred_dict.keys():
        print(pred_dict[key].shape)
    with open(os.path.join(out_folder, 'cmr_results.pkl'), 'wb') as f:
        pickle.dump(pred_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--in_folder', type=str, required=True,
                        help='Path to input frames folder.')
    parser.add_argument('--out_folder', type=str, default=None,
                        help='Folder to save predictions pickle in')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    predict_on_frames(args)
