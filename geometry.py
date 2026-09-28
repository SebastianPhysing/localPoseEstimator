# contains helper functions for triangulate and visualization of the poses and the camera frustums
# This script is partly written with the help of AI.

import os
import json
import itertools
import warnings
import numpy as np
import cv2


# COCO-17 keypoints in YOLO-Pose order
KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
             "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
             "l_knee", "r_knee", "l_ankle", "r_ankle"]

# bones of the stick figure: (index of keypoint a, index of keypoint b)
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
         (0, 5), (0, 6)]


def quat_to_matrix(q):
    '''
    Converts a quaternion of COLMAP (w, x, y, z) into a 3x3 rotation matrix.

    :param q: quaternion [w, x, y, z]
    :return: rotation matrix (3x3)
    '''
    w, x, y, z = np.array(q) / np.linalg.norm(q)

    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    return R


def load_cameras(path):
    '''
    Reads calibration.json and computes for every camera the matrices we need later.

    :param path: path to calibration.json
    :return: dict camera name (N, NW, ...) --> dict with K, dist, R, t, P, center, size
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    with open(path) as f:
        calibration = json.load(f)["cameras"]

    cams = {}
    for name, c in calibration.items():
        R = quat_to_matrix(c["qvec"])
        t = np.array(c["tvec"])

        # intrinsic matrix
        K = np.array([[c["f"], 0, c["cx"]], [0, c["f"], c["cy"]], [0, 0, 1]])

        # distortion as in OpenCV (k1, k2, p1, p2), RADIAL has no p1, p2
        dist = np.array([c["k1"], c["k2"], 0, 0])

        # projection matrix for undistorted, normalized coordinates (therefore without K)
        P = np.hstack([R, t[:, None]])

        # camera center in world coordinates
        center = -R.T @ t

        cams[name] = {
            "K": K,
            "dist": dist,
            "R": R,
            "t": t,
            "P": P,
            "center": center,
            "size": (c["width"], c["height"]),
        }

    return cams


def triangulate(cams, keypoints, min_conf=0.5, min_views=2):
    '''
    Triangulates the 3D position of every joint in every frame.
    Every pair of cameras is triangulated with OpenCV, the result is the median over all pairs.
    So a single wrong view (e.g. left/right swapped) is outvoted.

    :param cams: list of cameras (from load_cameras), same order as the views in keypoints
    :param keypoints: 2D keypoints (T, V, 17, 3) --> x, y, confidence
    :param min_conf: min. confidence of a keypoint
    :param min_views: min. number of cameras that have to see a joint
    :return: 3D points (T, 17, 3), NaN if the joint was seen by less than min_views cameras
    '''
    num_frames = keypoints.shape[0]
    num_views = keypoints.shape[1]
    num_joints = keypoints.shape[2]

    # keypoints with high enough confidence (T, V, 17)
    valid = keypoints[..., 2] >= min_conf

    # undistort all keypoints and convert them to normalized coordinates (without K)
    normalized = []
    for v in range(num_views):
        # NaN --> 0, these keypoints are not valid anyway
        pixels = np.nan_to_num(keypoints[:, v, :, :2])
        pixels = pixels.reshape(-1, 1, 2).astype(np.float64)
        undistorted = cv2.undistortPoints(pixels, cams[v]["K"], cams[v]["dist"])
        normalized.append(undistorted.reshape(-1, 2).T)  # (2, T*17)

    # triangulate every pair of cameras
    points_all_pairs = []
    for a, b in itertools.combinations(range(num_views), 2):
        X = cv2.triangulatePoints(cams[a]["P"], cams[b]["P"], normalized[a], normalized[b])

        # homogeneous --> 3D
        X = X[:3] / X[3]
        X = X.T.reshape(num_frames, num_joints, 3)

        # only keep joints that both cameras have seen
        both_valid = valid[:, a] & valid[:, b]
        X[~both_valid] = np.nan

        points_all_pairs.append(X)

    # median over all pairs. If all pairs are NaN the result is NaN, the warning for this is ignored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        points = np.nanmedian(points_all_pairs, axis=0)

    # number of cameras that have seen the joint
    num_seen = valid.sum(axis=1)
    points[num_seen < min_views] = np.nan

    return points


def reprojection_error(cam, points, keypoints, min_conf=0.5):
    '''
    Pixel distance between the detected joints and the 3D points projected back into the image of one camera.

    :param cam: camera (from load_cameras)
    :param points: 3D points (T, 17, 3)
    :param keypoints: 2D keypoints of this camera (T, 17, 3)
    :param min_conf: min. confidence of a keypoint
    :return: errors in pixel of all joints this camera has seen (1D array)
    '''
    # joints that the camera has seen and that could be triangulated
    used = (keypoints[..., 2] >= min_conf) & ~np.isnan(points[..., 0])
    if not used.any():
        return np.array([])

    # project the 3D points into the image (with distortion)
    rvec, _ = cv2.Rodrigues(cam["R"])
    projected, _ = cv2.projectPoints(points[used], rvec, cam["t"], cam["K"], cam["dist"])
    projected = projected.reshape(-1, 2)

    detected = keypoints[used][:, :2]
    errors = np.linalg.norm(projected - detected, axis=1)

    return errors


def to_display(points, cams):
    '''
    Converts world coordinates into the coordinates of the viewer:
    origin in the middle of the camera ring, y up (mean image-up direction of all cameras), z towards camera S.

    :param points: 3D points in world coordinates (..., 3)
    :param cams: dict of all cameras (from load_cameras)
    :return: 3D points in viewer coordinates (..., 3)
    '''
    # middle of all camera centers
    centers = [c["center"] for c in cams.values()]
    origin = np.mean(centers, axis=0)

    # image up = minus y axis of the camera (second row of R)
    ups = [-c["R"][1] for c in cams.values()]
    up = np.mean(ups, axis=0)
    up = up / np.linalg.norm(up)

    # direction to camera S, made orthogonal to up
    forward = cams["S"]["center"] - origin
    forward = forward - (forward @ up) * up
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)

    # rotation matrix, rows = new x, y, z axis
    M = np.stack([right, up, forward])

    return (points - origin) @ M.T


def pyramid(cam, depth):
    '''
    Camera center and the 4 image corners at the given depth (world coordinates).
    Used to draw the cameras as pyramids in the viewer.

    :param cam: camera (from load_cameras)
    :param depth: distance of the corners to the camera center
    :return: (5, 3) array: camera center, 4 corners
    '''
    w, h = cam["size"]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)

    # pixel --> ray direction in camera coordinates (length 1)
    rays = corners @ np.linalg.inv(cam["K"]).T
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    # rays @ R = rotate the rays from camera into world coordinates
    corners_world = cam["center"] + depth * rays @ cam["R"]

    return np.vstack([cam["center"], corners_world])