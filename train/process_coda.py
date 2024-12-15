import os
import pickle
from PIL import Image
import argparse
import tqdm
import numpy as np
import glob

from scipy.spatial.transform import Rotation as R

def parse_pose_file(pose_file):
    """
    Parse the poses from a given txt file.
    Expected format (per line):
    timestamp x y yaw
    or just x y yaw
    If there's a timestamp, we skip it.
    If your data format differs, adjust accordingly.
    """
    pose_np = np.loadtxt(pose_file, dtype=np.float64)

    positions = pose_np[:, 1:3].astype(np.float32)
    quat = R.from_quat(pose_np[:, [5,6,7,4]]).as_euler("zyx")
    yaws = quat[:, 0].astype(np.float32)

    return positions, yaws

def find_image_files(input_dir, sequence):
    """
    Find all images in 2d_raw/cam0 for a given sequence.
    The pattern for CODa might look like:
    2d_raw/cam0/cam0_{SEQUENCE}_{FRAME}.png
    We'll use glob to find them.

    Returns a list of tuples: (frame_index, filepath)
    """
    pattern = os.path.join(input_dir, "2d_raw", "cam0", str(sequence), f"2d_raw_cam0_{sequence}_*.jpg")
    files = glob.glob(pattern)
    # Extract frame number from filename (assuming the last underscore splits sequence and frame)
    # Example filename: cam1_0001_23.jpg -> sequence=0001, frame=23
    # We'll split by '_' and take the last part before '.png' as frame
    frame_file_pairs = []
    for fpath in files:
        fname = os.path.basename(fpath)
        # fname like cam1_{sequence}_{frame}.png
        # split by '_' and pick last before extension
        parts = fname.split('_')
        frame_str = parts[-1].replace(".jpg", "")
        frame_idx = int(frame_str)
        frame_file_pairs.append((frame_idx, fpath))
    # Sort by frame index
    frame_file_pairs.sort(key=lambda x: x[0])
    return frame_file_pairs

def main(args: argparse.Namespace):
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all pose files in poses/dense_global
    pose_dir = os.path.join(input_dir, "poses", "dense")
    pose_files = glob.glob(os.path.join(pose_dir, "*.txt"))

    # Extract sequence identifiers from these files
    # Assuming filenames like: XXX.txt where XXX is the sequence number
    sequences = []
    for p in pose_files:
        base = os.path.basename(p)
        seq = os.path.splitext(base)[0]
        sequences.append(seq)

    # If num_trajs is specified, slice the list
    if args.num_trajs >= 0:
        sequences = sequences[: args.num_trajs]

    for seq in tqdm.tqdm(sequences, desc="Trajectories processed"):
        # Parse pose file
        pose_file = os.path.join(pose_dir, f"{seq}.txt")
        positions, yaws = parse_pose_file(pose_file)

        # Create trajectory folder
        traj_folder = os.path.join(output_dir, seq)
        os.makedirs(traj_folder, exist_ok=True)

        # Save traj_data.pkl
        traj_data = {"position": positions, "yaw": yaws}
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        # Find images in 2d_raw/cam0 for this sequence
        frame_file_pairs = find_image_files(input_dir, seq)
        
        # Check if the number of images matches the number of poses
        # If they differ, we will use the min count so we don't run out of images or poses
        count = min(len(frame_file_pairs), len(positions))
        for i in range(count):
            frame_idx, fpath = frame_file_pairs[i]
            img = Image.open(fpath)
            img.convert("RGB").save(os.path.join(traj_folder, f"{i}.jpg"))
        
        # If there's a mismatch, warn the user
        if len(frame_file_pairs) != len(positions):
            print(f"Warning: sequence {seq} has {len(frame_file_pairs)} images but {len(positions)} poses. Using min count {count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path to the CREStE dataset root directory",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/scratch/arthurz/Datasets/vint/creste/",
        type=str,
        help="path for processed dataset (default: /scratch/arthurz/Datasets/vint/creste/)",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of trajectories to process (default: -1, all)",
    )

    args = parser.parse_args()
    print("STARTING PROCESSING CREStE DATASET")
    main(args)
    print("FINISHED PROCESSING CREStE DATASET")
