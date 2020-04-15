import torch
import numpy as np


def orthographic_project_torch(points3D, cam_params):
    """
    Scaled orthographic projection (i.e. weak perspective projection).
    Should be going from SMPL 3D coords to [-1, 1] scaled image coords.
    cam_params are [s, tx, ty]  - i.e. scaling and 2D translation.
    """
    x = points3D[:, :, 0]
    y = points3D[:, :, 1]

    # Scaling
    s = torch.unsqueeze(cam_params[:, 0], dim=1)

    # Translation
    t_x = torch.unsqueeze(cam_params[:, 1], dim=1)
    t_y = torch.unsqueeze(cam_params[:, 2], dim=1)

    u = s * (x + t_x)
    v = s * (y + t_y)

    proj_points = torch.stack([u, v], dim=-1)

    return proj_points


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def get_intrinsics_matrix(img_width, img_height, focal_length):
    """
    Camera intrinsic matrix (calibration matrix) given focal length and img_width and
    img_height. Assumes that principal point is at (width/2, height/2).
    """
    K = np.array([[focal_length, 0., img_width/2.0],
                  [0., focal_length, img_height/2.0],
                  [0., 0., 1.]])
    return K


def convert_weak_perspective_to_camera_translation(cam, focal_length, resolution):
    cam_t = np.array([cam[1], cam[2], 2 * focal_length / (resolution * cam[0] + 1e-9)])
    return cam_t


def batch_convert_weak_perspective_to_camera_translation(wp_cams, focal_length, resolution):
    num = wp_cams.shape[0]
    cam_ts = np.zeros((num, 3), dtype=np.float32)
    for i in range(num):
        cam_t = convert_weak_perspective_to_camera_translation(wp_cams[i],
                                                               focal_length,
                                                               resolution)
        cam_ts[i] = cam_t.astype(np.float32)
    return cam_ts