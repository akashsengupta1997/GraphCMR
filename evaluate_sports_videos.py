import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2

import config
from utils import Mesh
from models import CMR, NMRRenderer
from models.smpl_from_lib import SMPL
from utils.pose_utils import compute_similarity_transform_batch, scale_and_translation_transform_batch
from utils.cam_utils import orthographic_project_torch, undo_keypoint_normalisation, \
    get_intrinsics_matrix, batch_convert_weak_perspective_to_camera_translation
from utils.label_conversions import convert_multiclass_to_binary_labels_torch
from datasets.sports_videos_eval_dataset import SportsVideosEvalDataset


def evaluate_single_in_multitasknet_sports_videos(model,
                                                  eval_dataset,
                                                  metrics,
                                                  device,
                                                  save_path,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  vis_every_n_batches=1,
                                                  output_img_wh=256):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1)
    smpl.to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male')
    smpl_male.to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female')
    smpl_female.to(device)

    if 'pve' in metrics:
        pve_smpl_sum = 0.0
        pve_graph_sum = 0.0
        pve_smpl_per_frame = []
        pve_graph_per_frame = []

    if 'pve_scale_corrected' in metrics:
        pve_scale_corrected_smpl_sum = 0.0
        pve_scale_corrected_graph_sum = 0.0
        pve_scale_corrected_smpl_per_frame = []
        pve_scale_corrected_graph_per_frame = []

    if 'pve_pa' in metrics:
        pve_pa_smpl_sum = 0.0
        pve_pa_graph_sum = 0.0
        pve_pa_smpl_per_frame = []
        pve_pa_graph_per_frame = []

    if 'pve-t' in metrics:
        pvet_sum = 0.0
        pvet_per_frame = []

    if 'pve-t_scale_corrected' in metrics:
        pvet_scale_corrected_sum = 0.0
        pvet_scale_corrected_per_frame = []

    if 'silhouette_iou' in metrics:
        # Set-up NMR renderer to render silhouettes from predicted vertex meshes.
        # Assuming camera rotation is identity (since it is dealt with by global_orients)
        cam_K = get_intrinsics_matrix(output_img_wh, output_img_wh,
                                      config.FOCAL_LENGTH)
        cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
        cam_R = torch.eye(3).to(device)
        cam_K = cam_K[None, :, :]
        cam_R = cam_R[None, :, :]
        nmr_parts_renderer = NMRRenderer(1,
                                         cam_K,
                                         cam_R,
                                         output_img_wh,
                                         rend_parts_seg=True).to(device)
        num_true_positives_smpl = 0.0
        num_false_positives_smpl = 0.0
        num_true_negatives_smpl = 0.0
        num_false_negatives_smpl = 0.0
        num_true_positives_graph = 0.0
        num_false_positives_graph = 0.0
        num_true_negatives_graph = 0.0
        num_false_negatives_graph = 0.0
        silhouette_iou_smpl_per_frame = []
        silhouette_iou_graph_per_frame = []

    if 'j2d_l2e' in metrics:
        j2d_l2e_sum = 0.0
        j2d_l2e_per_frame = []

    num_samples = 0
    num_vertices = 6890
    num_joints2d = 17

    frame_path_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []

    model.eval()
    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input']
        input = input.to(device)

        target_shape = samples_batch['shape']
        target_shape = target_shape.to(device)
        target_vertices = samples_batch['vertices']
        target_silhouette = samples_batch['silhouette']
        target_joints2d_coco = samples_batch['keypoints']

        target_gender = samples_batch['gender'][0]
        if target_gender == 'm':
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_reposed_smpl_output = smpl_female(betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices
        # ------------------------------- PREDICTIONS -------------------------------
        pred_vertices, pred_vertices_smpl, pred_camera, pred_rotmat, pred_betas = model(input)
        pred_vertices_projected2d = orthographic_project_torch(pred_vertices, pred_camera)
        pred_vertices_projected2d = undo_keypoint_normalisation(pred_vertices_projected2d, input.shape[-1])
        pred_vertices_smpl_projected2d = orthographic_project_torch(pred_vertices_smpl, pred_camera)
        pred_vertices_smpl_projected2d = undo_keypoint_normalisation(pred_vertices_smpl_projected2d, input.shape[-1])
        pred_reposed_smpl_output = smpl(betas=pred_betas)
        pred_reposed_vertices = pred_reposed_smpl_output.vertices

        if 'j2d_l2e' in metrics:
            pred_smpl_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                    global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_camera)
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco, output_img_wh)

        pred_camera = pred_camera.cpu().detach().numpy()
        if 'silhouette_iou' in metrics:
            pred_cam_ts = batch_convert_weak_perspective_to_camera_translation(pred_camera,
                                                                               config.FOCAL_LENGTH,
                                                                               output_img_wh)
            pred_cam_ts = torch.from_numpy(pred_cam_ts).float().to(device)
            part_seg = nmr_parts_renderer(pred_vertices, pred_cam_ts.unsqueeze(0))
            pred_silhouette = convert_multiclass_to_binary_labels_torch(part_seg)
            pred_silhouette = pred_silhouette.cpu().detach().numpy()
            part_seg_smpl = nmr_parts_renderer(pred_vertices_smpl, pred_cam_ts.unsqueeze(0))
            pred_silhouette_smpl = convert_multiclass_to_binary_labels_torch(part_seg_smpl)
            pred_silhouette_smpl = pred_silhouette_smpl.cpu().detach().numpy()

        # Numpy-fying
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        pred_vertices = pred_vertices.cpu().detach().numpy()
        pred_vertices_smpl = pred_vertices_smpl.cpu().detach().numpy()
        pred_vertices_projected2d = pred_vertices_projected2d.cpu().detach().numpy()
        pred_vertices_smpl_projected2d = pred_vertices_smpl_projected2d.cpu().detach().numpy()
        pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
        pred_rotmat = pred_rotmat.cpu().detach().numpy()
        pred_betas = pred_betas.cpu().detach().numpy()

        # ------------------------------- METRICS -------------------------------
        if 'pve' in metrics:
            pve_smpl_batch = np.linalg.norm(pred_vertices_smpl - target_vertices,
                                            axis=-1)  # (1, 6890)
            pve_graph_batch = np.linalg.norm(pred_vertices - target_vertices,
                                             axis=-1)
            pve_smpl_sum += np.sum(pve_smpl_batch)  # scalar
            pve_graph_sum += np.sum(pve_graph_batch)
            pve_smpl_per_frame.append(np.mean(pve_smpl_batch, axis=-1))
            pve_graph_per_frame.append(np.mean(pve_graph_batch, axis=-1))

        # Scale and translation correction
        if 'pve_scale_corrected' in metrics:
            pred_vertices_smpl_sc = scale_and_translation_transform_batch(pred_vertices_smpl,
                                                                          target_vertices)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pve_sc_smpl_batch = np.linalg.norm(pred_vertices_smpl_sc - target_vertices,
                                               axis=-1)  # (1, 6890)
            pve_sc_graph_batch = np.linalg.norm(pred_vertices_sc - target_vertices,
                                                axis=-1)  # (1, 6890)
            pve_scale_corrected_smpl_sum += np.sum(pve_sc_smpl_batch)  # scalar
            pve_scale_corrected_graph_sum += np.sum(pve_sc_graph_batch)  # scalar
            pve_scale_corrected_smpl_per_frame.append(np.mean(pve_sc_smpl_batch, axis=-1))
            pve_scale_corrected_graph_per_frame.append(np.mean(pve_sc_graph_batch, axis=-1))

        # Procrustes analysis
        if 'pve_pa' in metrics:
            pred_vertices_smpl_pa = compute_similarity_transform_batch(pred_vertices_smpl, target_vertices)
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices, target_vertices)
            pve_pa_smpl_batch = np.linalg.norm(pred_vertices_smpl_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_graph_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (1, 6890)
            pve_pa_smpl_sum += np.sum(pve_pa_smpl_batch)  # scalar
            pve_pa_graph_sum += np.sum(pve_pa_graph_batch)  # scalar
            pve_pa_smpl_per_frame.append(np.mean(pve_pa_smpl_batch, axis=-1))
            pve_pa_graph_per_frame.append(np.mean(pve_pa_graph_batch, axis=-1))

        if 'pve-t' in metrics:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices, axis=-1)
            pvet_sum += np.sum(pvet_batch)
            pvet_per_frame.append(np.mean(pvet_batch, axis=-1))

        # Scale and translation correction
        if 'pve-t_scale_corrected' in metrics:
            pred_reposed_vertices_sc = compute_similarity_transform_batch(pred_reposed_vertices,
                                                                          target_reposed_vertices)
            pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices, axis=-1)  # (1, 6890)
            pvet_scale_corrected_sum += np.sum(pvet_sc_batch)  # scalar
            pvet_scale_corrected_per_frame.append(np.mean(pvet_sc_batch, axis=-1))

        if 'silhouette_iou' in metrics:
            pred_silhouette = np.round(pred_silhouette).astype(np.bool)
            target_silhouette = np.round(target_silhouette).astype(np.bool)

            true_positive = np.logical_and(pred_silhouette, target_silhouette)
            false_positive = np.logical_and(pred_silhouette,
                                            np.logical_not(target_silhouette))
            true_negative = np.logical_and(np.logical_not(pred_silhouette),
                                           np.logical_not(target_silhouette))
            false_negative = np.logical_and(np.logical_not(pred_silhouette),
                                            target_silhouette)

            num_tp = int(np.sum(true_positive))
            num_fp = int(np.sum(false_positive))
            num_tn = int(np.sum(true_negative))
            num_fn = int(np.sum(false_negative))

            num_true_positives_graph += num_tp
            num_false_positives_graph += num_fp
            num_true_negatives_graph += num_tn
            num_false_negatives_graph += num_fn

            silhouette_iou_graph_per_frame.append(num_tp / (num_tp + num_fp + num_fn))

            pred_silhouette_smpl = np.round(pred_silhouette_smpl).astype(np.bool)
            target_silhouette = np.round(target_silhouette).astype(np.bool)

            true_positive = np.logical_and(pred_silhouette_smpl, target_silhouette)
            false_positive = np.logical_and(pred_silhouette_smpl,
                                            np.logical_not(target_silhouette))
            true_negative = np.logical_and(np.logical_not(pred_silhouette_smpl),
                                           np.logical_not(target_silhouette))
            false_negative = np.logical_and(np.logical_not(pred_silhouette_smpl),
                                            target_silhouette)

            num_tp = int(np.sum(true_positive))
            num_fp = int(np.sum(false_positive))
            num_tn = int(np.sum(true_negative))
            num_fn = int(np.sum(false_negative))

            num_true_positives_smpl += num_tp
            num_false_positives_smpl += num_fp
            num_true_negatives_smpl += num_tn
            num_false_negatives_smpl += num_fn

            silhouette_iou_smpl_per_frame.append(num_tp / (num_tp + num_fp + num_fn))

        if 'j2d_l2e' in metrics:
            j2d_l2e_batch = np.linalg.norm(pred_joints2d_coco - target_joints2d_coco,
                                           axis=-1)  # (bs, 17)
            j2d_l2e_sum += np.sum(j2d_l2e_batch)  # scalar
            j2d_l2e_per_frame.append(np.mean(j2d_l2e_batch, axis=-1))

        num_samples += target_shape.shape[0]

        frame_path = samples_batch['frame_path']
        frame_path_per_frame .append(frame_path)
        pose_per_frame.append(pred_rotmat)
        shape_per_frame.append(pred_betas)
        cam_per_frame.append(pred_camera)

        # ------------------------------- VISUALISE -------------------------------
        if batch_num % vis_every_n_batches == 0:
            vis_imgs = samples_batch['vis_img'].numpy()
            vis_imgs = np.transpose(vis_imgs, [0, 2, 3, 1])

            plt.figure(figsize=(12, 8))
            plt.subplot(341)
            plt.imshow(vis_imgs[0])

            plt.subplot(342)
            plt.imshow(vis_imgs[0])
            plt.scatter(pred_vertices_projected2d[0, :, 0], pred_vertices_projected2d[0, :, 1], s=0.1, c='r')

            plt.subplot(343)
            plt.imshow(vis_imgs[0])
            plt.scatter(pred_vertices_smpl_projected2d[0, :, 0], pred_vertices_smpl_projected2d[0, :, 1], s=0.1, c='r')

            if 'silhouette_iou' in metrics:
                plt.subplot(344)
                plt.imshow(pred_silhouette[0].astype(np.int16) -
                           target_silhouette[0].astype(np.int16))
                plt.subplot(345)
                plt.imshow(pred_silhouette_smpl[0].astype(np.int16) -
                           target_silhouette[0].astype(np.int16))
                if 'j2d_l2e' in metrics:
                    for j in range(target_joints2d_coco.shape[1]):
                        plt.scatter(target_joints2d_coco[0, j, 0],
                                    target_joints2d_coco[0, j, 1],
                                    c='b', s=10.0)
                        plt.text(target_joints2d_coco[0, j, 0], target_joints2d_coco[0, j, 1],
                                 str(j))
                        plt.scatter(pred_joints2d_coco[0, j, 0],
                                    pred_joints2d_coco[0, j, 1],
                                    c='r', s=10.0)
                        plt.text(pred_joints2d_coco[0, j, 0], pred_joints2d_coco[0, j, 1],
                                 str(j))

            plt.subplot(346)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices[0, :, 0], pred_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(347)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices_smpl[0, :, 0], pred_vertices_smpl[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(348)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices_pa[0, :, 0], pred_vertices_pa[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(349)
            plt.scatter(target_vertices[0, :, 0], target_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_vertices_smpl_pa[0, :, 0], pred_vertices_smpl_pa[0, :, 1], s=0.05, c='r')
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(3, 4, 10)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices[0, :, 0], pred_reposed_vertices[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            plt.subplot(3, 4, 11)
            plt.scatter(target_reposed_vertices[0, :, 0], target_reposed_vertices[0, :, 1], s=0.1, c='b')
            plt.scatter(pred_reposed_vertices_sc[0, :, 0], pred_reposed_vertices_sc[0, :, 1], s=0.05, c='r')
            plt.gca().set_aspect('equal', adjustable='box')

            # plt.show()
            split_path = frame_path[0].split('/')
            clip_name = split_path[-3]
            frame_num = split_path[-1]
            save_fig_path = os.path.join(save_path, clip_name + '_' + frame_num)
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

    # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    frame_path_per_frame = np.concatenate(frame_path_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), frame_path_per_frame)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)

    if 'pve' in metrics:
        pve_smpl = pve_smpl_sum / (num_samples * num_vertices)
        print('PVE SMPL: {:.5f}'.format(pve_smpl))
        pve_graph = pve_graph_sum / (num_samples * num_vertices)
        print('PVE GRAPH: {:.5f}'.format(pve_graph))
        pve_smpl_per_frame = np.concatenate(pve_smpl_per_frame, axis=0)
        pve_graph_per_frame = np.concatenate(pve_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_per_frame.npy'), pve_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_graph_per_frame.npy'), pve_graph_per_frame)

    if 'pve_scale_corrected' in metrics:
        pve_sc_smpl = pve_scale_corrected_smpl_sum / (num_samples * num_vertices)
        print('PVE SC SMPL: {:.5f}'.format(pve_sc_smpl))
        pve_sc_graph = pve_scale_corrected_graph_sum / (num_samples * num_vertices)
        print('PVE SC GRAPH: {:.5f}'.format(pve_sc_graph))
        pve_scale_corrected_smpl_per_frame = np.concatenate(pve_scale_corrected_smpl_per_frame,
                                                            axis=0)
        pve_scale_corrected_graph_per_frame = np.concatenate(pve_scale_corrected_graph_per_frame,
                                                             axis=0)
        np.save(os.path.join(save_path, 'pve_scale_corrected_per_frame.npy'), pve_scale_corrected_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_scale_corrected_graph_per_frame.npy'), pve_scale_corrected_graph_per_frame)

    if 'pve_pa' in metrics:
        pve_pa_smpl = pve_pa_smpl_sum / (num_samples * num_vertices)
        print('PVE PA SMPL: {:.5f}'.format(pve_pa_smpl))
        pve_pa_graph = pve_pa_graph_sum / (num_samples * num_vertices)
        print('PVE PA GRAPH: {:.5f}'.format(pve_pa_graph))
        pve_pa_smpl_per_frame = np.concatenate(pve_pa_smpl_per_frame, axis=0)
        pve_pa_graph_per_frame = np.concatenate(pve_pa_graph_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pve_pa_per_frame.npy'), pve_pa_smpl_per_frame)
        np.save(os.path.join(save_path, 'pve_pa_graph_per_frame.npy'), pve_pa_graph_per_frame)

    if 'pve-t' in metrics:
        pvet = pvet_sum / (num_samples * num_vertices)
        print('PVE-T: {:.5f}'.format(pvet))
        pvet_per_frame = np.concatenate(pvet_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_per_frame.npy'), pvet_per_frame)

    if 'pve-t_scale_corrected' in metrics:
        pvet_sc = pvet_scale_corrected_sum / (num_samples * num_vertices)
        print('PVE-T SC: {:.5f}'.format(pvet_sc))
        pvet_scale_corrected_per_frame = np.concatenate(pvet_scale_corrected_per_frame, axis=0)
        np.save(os.path.join(save_path, 'pvet_scale_corrected_per_frame.npy'), pvet_scale_corrected_per_frame)

    if 'silhouette_iou' in metrics:
        mean_iou_graph = num_true_positives_graph / (
                num_true_positives_graph + num_false_negatives_graph + num_false_positives_graph)
        global_acc_graph = (num_true_positives_graph + num_true_negatives_graph) / (
                num_true_positives_graph + num_true_negatives_graph + num_false_negatives_graph + num_false_positives_graph)
        mean_iou_smpl = num_true_positives_smpl / (
                num_true_positives_smpl + num_false_negatives_smpl + num_false_positives_smpl)
        global_acc_smpl = (num_true_positives_smpl + num_true_negatives_smpl) / (
                num_true_positives_smpl + num_true_negatives_smpl + num_false_negatives_smpl + num_false_positives_smpl)
        np.save(os.path.join(save_path, 'silhouette_iou_per_frame.npy'),
                silhouette_iou_smpl_per_frame)
        np.save(os.path.join(save_path, 'silhouette_iou_graph_per_frame.npy'),
                silhouette_iou_graph_per_frame)
        print('Mean IOU SMPL: {:.3f}'.format(mean_iou_smpl))
        print('Global Acc SMPL: {:.3f}'.format(global_acc_smpl))
        print('Mean IOU Graph: {:.3f}'.format(mean_iou_graph))
        print('Global Acc Graph: {:.3f}'.format(global_acc_graph))

    if 'j2d_l2e' in metrics:
        j2d_l2e = j2d_l2e_sum / (num_samples * num_joints2d)
        j2d_l2e_per_frame = np.concatenate(j2d_l2e_per_frame, axis=0)
        np.save(os.path.join(save_path, 'j2d_l2e_per_frame_per_frame.npy'),
                j2d_l2e_per_frame)
        print('J2D L2 Error: {:.5f}'.format(j2d_l2e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
    parser.add_argument('--path_correction', action='store_true')
    args = parser.parse_args()

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    mesh = Mesh(device=device)
    # Our pretrained networks have 5 residual blocks with 256 channels.
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint, device=device)
    model.to(device)
    model.eval()

    # Setup evaluation dataset
    dataset_path = '/scratch/as2562/datasets/sports_videos_smpl/final_dataset'
    dataset = SportsVideosEvalDataset(dataset_path, img_wh=config.INPUT_RES,
                                      path_correction=args.path_correction)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pve', 'pve_scale_corrected', 'pve_pa', 'pve-t', 'pve-t_scale_corrected',
               'silhouette_iou', 'j2d_l2e']

    save_path = '/data/cvfs/as2562/GraphCMR/evaluations/sports_videos_final_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Run evaluation
    evaluate_single_in_multitasknet_sports_videos(model,
                                                  dataset,
                                                  metrics,
                                                  device,
                                                  save_path,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True,
                                                  vis_every_n_batches=1)




