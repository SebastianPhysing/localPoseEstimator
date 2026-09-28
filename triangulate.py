# triangulates the 3d points given of the extracted 2d points of multiple views

# This code is partilly written with the help of AI.

import os
import json
import argparse
import numpy as np
import geometry


CALIBRATION = "calibration.json"  # has to be in the folder where you run this script

# older recordings use the ip number (nuc12), newer ones the position (nucS)
NUMBER_TO_CAMERA = {"12": "S", "13": "N", "14": "NW", "15": "SW", "16": "NE", "17": "SE"}


def get_camera_names(folders):
    '''
    Converts the folder names (nucS, nuc12, ...) into the camera names of calibration.json (S, N, NW, ...).

    :param folders: list of folder names
    :return: list of camera names
    '''
    names = []
    for folder in folders:
        name = folder.removeprefix("nuc")
        if name in NUMBER_TO_CAMERA:
            name = NUMBER_TO_CAMERA[name]
        names.append(name)

    return names


def print_reprojection_errors(views, points, keypoints, folders, names):
    '''
    Prints the median reprojection error of every camera.

    :param views: list of cameras, same order as in keypoints
    :param points: 3D points (T, 17, 3)
    :param keypoints: 2D keypoints (T, V, 17, 3)
    :param folders: folder names (nucS, ...)
    :param names: camera names (S, ...)
    '''
    print("reprojection error per camera (median):")
    for v in range(len(views)):
        errors = geometry.reprojection_error(views[v], points, keypoints[:, v])
        if len(errors) == 0:
            print(f"  {folders[v]:6s} {names[v]:3s}  no joints")
        else:
            print(f"  {folders[v]:6s} {names[v]:3s} {np.median(errors):6.2f} px  ({len(errors)} joints)")


def make_viewer_data(points, cams, times):
    '''
    Puts everything the viewer needs into one dict (is saved as poses3d.json).

    :param points: 3D points (T, 17, 3) in world coordinates
    :param cams: dict of all cameras (from load_cameras)
    :param times: time of every frame in seconds
    :return: dict with keypoints, bones, size, focus, cameras, times, frames
    '''
    display = geometry.to_display(points, cams)

    # radius of the camera ring, used as scale in the viewer
    centers = []
    for c in cams.values():
        centers.append(c["center"])
    centers = geometry.to_display(np.array(centers), cams)
    size = float(np.mean(np.linalg.norm(centers, axis=1)))

    # cameras are drawn as pyramids
    cameras = []
    for name, c in cams.items():
        pyramid = geometry.pyramid(c, 0.12 * size)
        cameras.append({"name": name, "pyramid": geometry.to_display(pyramid, cams).tolist()})

    # 3D points of every frame, None if the joint could not be triangulated
    frames = []
    for frame in display:
        joints = []
        for p in frame:
            if np.isnan(p[0]):
                joints.append(None)
            else:
                joints.append(p.round(4).tolist())
        frames.append(joints)

    viewer = {
        "keypoints": geometry.KEYPOINTS,
        "bones": geometry.BONES,
        "size": size,
        "focus": np.nanmean(display, axis=(0, 1)).tolist(),  # middle of all joints, the viewer looks at it
        "cameras": cameras,
        "times": times.round(3).tolist(),
        "frames": frames,
    }

    return viewer


def triangulate_recording(synced_dir, min_views):
    '''
    Triangulates the 3D skeleton of one recording and saves poses3d.json.

    :param synced_dir: synced folder (e.g. recordings/test/synced)
    :param min_views: min. number of cameras that have to see a joint
    '''
    print('------------------ TRIANGULATION -------------------')

    npz_path = os.path.join(synced_dir, "keypoints2d.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Run detect_2d.py first.")

    data = np.load(npz_path)
    keypoints = data["keypoints"]
    cams = geometry.load_cameras(CALIBRATION)

    folders = [str(name) for name in data["cameras"]]
    names = get_camera_names(folders)

    # cameras in the same order as in keypoints2d.npz
    views = []
    for name in names:
        views.append(cams[name])

    points = geometry.triangulate(views, keypoints, min_views=min_views)
    print(f"triangulated {np.mean(~np.isnan(points[..., 0])) * 100:.1f} % of all joints")

    print_reprojection_errors(views, points, keypoints, folders, names)

    viewer = make_viewer_data(points, cams, data["times"])

    out_path = os.path.join(synced_dir, "poses3d.json")
    with open(out_path, "w") as f:
        json.dump(viewer, f)

    print(f"poses3d.json has been written to {out_path}")
    print('----------------------------------------------------')


def main():
    parser = argparse.ArgumentParser(description="Triangulate the 3D skeleton")
    parser.add_argument("synced_dir", type=str, nargs="?", help="e.g. recordings/test/synced")
    parser.add_argument("--min-views", type=int, default=2, help="min. cameras per joint (default: 2)")
    args = parser.parse_args()

    # Interactive fallback
    if args.synced_dir is None:
        args.synced_dir = input("Synced folder (e.g. recordings/test/synced): ").strip()

    triangulate_recording(args.synced_dir, args.min_views)


# python3 triangulate.py recordings/test/synced
# python3 triangulate.py recordings/test/synced --min-views 3

if __name__ == "__main__":
    main()