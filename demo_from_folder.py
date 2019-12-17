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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
parser.add_argument('--in_folder', type=str, required=True, help='Path to input images folder.')
parser.add_argument('--out_folder', type=str, default=None,
                    help='Folder to save predictions in')


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


def bbox_from_pkl(bbox_file):
    with open(bbox_file, 'rb') as f:
        bbox = np.array(pickle.load(f)[0]).astype(np.float32)
        print(bbox)

    ul_corner = bbox[:2]
    # Getting bbox into the expected format...
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    bbox[2:] = [width, height]
    bbox[[0, 1]] = bbox[[1, 0]]

    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    img = cv2.imread(img_file)[:, :,
          ::-1].copy()  # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_pkl(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])])
    # plt.show()

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def write_ply_file(fpath, verts, colour):
    ply_header = '''ply
                    format ascii 1.0
                    element vertex {}
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                   '''
    num_verts = verts.shape[0]
    color_array = np.tile(np.array(colour), (num_verts, 1))
    verts_with_colour = np.concatenate([verts, color_array], axis=-1)
    with open(fpath, 'w') as f:
        f.write(ply_header.format(num_verts))
        np.savetxt(f, verts_with_colour, '%f %f %f %d %d %d')


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    mesh = Mesh(device=device)
    # Our pretrained networks have 5 residual blocks with 256 channels.
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint, device=device)

    model.to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer()

    image_paths = [f for f in args.in_folder if f.endswith('.png')]
    print('Predicting on all png images in {}'.format(args.in_folder))

    for image_path in image_paths:
        # Preprocess input image and generate predictions
        bbox_path = os.path.splitext(image_path)[0] + '_bb_coords.pkl'
        assert os.path.exists(bbox_path), "Bounding boxes required for {}!".format(image_path)
        img, norm_img = process_image(image_paths, bbox_path, None, input_res=cfg.INPUT_RES)
        norm_img = norm_img.to(device)
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, pred_camera, _, _ = model(norm_img)

        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                          2 * cfg.FOCAL_LENGTH / (
                                                      cfg.INPUT_RES * pred_camera[:, 0] + 1e-9)],
                                         dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        pred_vertices_smpl = pred_vertices_smpl[0].cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()

        # Plot and save results
        # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
        outfile = os.path.join(args.out_folder,
                               os.path.splitext(os.path.basename(args.img))[0])
        print('Saving to:', outfile)

        plt.figure()
        plt.axis('off')
        plt.tight_layout()

        subplot_count = 1
        # plot image
        plt.subplot(1, 3, subplot_count)
        plt.imshow(np.squeeze(img))
        subplot_count += 1

        # plot GCN predicted verts
        plt.subplot(1, 3, subplot_count)
        plt.scatter(pred_vertices[:, 0],
                    pred_vertices[:, 1],
                    s=0.6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        subplot_count += 1

        # plot SMPL predicted verts
        plt.subplot(1, 3, subplot_count)
        plt.scatter(pred_vertices_smpl[:, 0],
                    pred_vertices_smpl[:, 1],
                    s=0.6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        subplot_count += 1
        plt.savefig(outfile + "_verts_plot.png", bbox_inches='tight')
        # plt.show()

        # Render non-parametric shape
        img_gcnn = renderer.render(pred_vertices, mesh.faces.cpu().numpy(),
                                   camera_t=camera_translation,
                                   img=img, use_bg=True, body_color='pink')

        # Render parametric shape
        img_smpl = renderer.render(pred_vertices_smpl, mesh.faces.cpu().numpy(),
                                   camera_t=camera_translation,
                                   img=img, use_bg=True, body_color='light_blue')

        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        center_smpl = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        rot_vertices_smpl = np.dot((pred_vertices_smpl - center_smpl), aroundy) + center_smpl

        # Render non-parametric shape
        img_gcnn_side = renderer.render(rot_vertices, mesh.faces.cpu().numpy(),
                                        camera_t=camera_translation,
                                        img=np.ones_like(img), use_bg=True, body_color='pink')

        # Render parametric shape
        img_smpl_side = renderer.render(rot_vertices_smpl, mesh.faces.cpu().numpy(),
                                        camera_t=camera_translation,
                                        img=np.ones_like(img), use_bg=True,
                                        body_color='light_blue')

        # Save reconstructions
        cv2.imwrite(outfile + '_gcnn.png', 255 * img_gcnn[:, :, ::-1])
        cv2.imwrite(outfile + '_smpl.png', 255 * img_smpl[:, :, ::-1])
        cv2.imwrite(outfile + '_gcnn_side.png', 255 * img_gcnn_side[:, :, ::-1])
        cv2.imwrite(outfile + '_smpl_side.png', 255 * img_smpl_side[:, :, ::-1])
        write_ply_file(outfile + '_gcn_verts.ply', pred_vertices, [255, 0, 0])
        write_ply_file(outfile + '_smpl_verts.ply', pred_vertices_smpl, [255, 0, 0])
