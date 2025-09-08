"""
rlds_action_discretization.py

This script performs a one-time calculation of action bin boundaries from an RLDS dataset.
It iterates through the entire dataset, collects all action vectors, computes percentile-based
bins for each action dimension, and saves the resulting bin boundaries to a NumPy file.

This pre-computation is necessary for discretizing continuous robot actions into a format
that a language model can predict.
"""

import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import argparse
import glob
import os
import tensorflow as tf

import json

# Disable noisy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

def calculate_and_save_action_bins(dataset_base_path: str, num_bins: int, output_path: str, info_output_path: str):
    """
    Calculates percentile-based action bins from multiple RLDS datasets and saves them to a file.
    Also counts the total number of steps and saves that info.

    Args:
        dataset_base_path (str): The base path containing the RLDS dataset directories.
        num_bins (int): The number of discrete bins to create for each action dimension.
        output_path (str): The path to save the calculated action bins (.npy file).
        info_output_path (str): The path to save the dataset info (e.g., total steps).
    """
    dataset_paths = glob.glob(f"{dataset_base_path}/libero_*_no_noops/1.0.0")
    if not dataset_paths:
        print(f"Error: No datasets found matching the pattern '{dataset_base_path}/libero_*_no_noops/1.0.0'.")
        return

    print(f"Found {len(dataset_paths)} datasets to process:")
    for path in dataset_paths:
        print(f"  - {path}")

    datasets = []
    for path in dataset_paths:
        try:
            builder = tfds.builder_from_directory(builder_dir=path)
            
            # Check what splits are available
            split_info = builder.info.splits
            available_splits = list(split_info.keys())
            print(f"\nDataset: {os.path.basename(path)}")
            print(f"  Available splits: {available_splits}")
            
            # Print detailed info for each split
            for split_name in available_splits:
                split_stats = split_info[split_name]
                print(f"    {split_name}: {split_stats.num_examples} examples")
            
            ds = builder.as_dataset(split='train')
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Could not load dataset from {path}. Error: {e}")
            continue
    
    if not datasets:
        print("Error: No valid datasets could be loaded.")
        return

    # First, calculate individual dataset lengths by processing each dataset separately
    dataset_steps = []
    print("\nCalculating individual dataset lengths...")
    for i, path in enumerate(dataset_paths):
        try:
            builder = tfds.builder_from_directory(builder_dir=path)
            ds = builder.as_dataset(split='train')
            
            # Count steps in this dataset
            dataset_step_count = 0
            for episode in ds:
                for step in episode['steps'].as_numpy_iterator():
                    dataset_step_count += 1
            
            dataset_steps.append(dataset_step_count)
            print(f"  {os.path.basename(path)}: {dataset_step_count} steps")
        except Exception as e:
            print(f"Warning: Could not process dataset {path}. Error: {e}")
            dataset_steps.append(0)

    # Create a single, concatenated dataset from all sources for action collection
    if len(datasets) == 1:
        concatenated_dataset = datasets[0]
    else:
        concatenated_dataset = datasets[0]
        for ds in datasets[1:]:
            concatenated_dataset = concatenated_dataset.concatenate(ds)

    print("\nCollecting all actions from the combined dataset. This may take a while...")
    all_actions = []
    total_steps = 0
    
    for episode in tqdm(concatenated_dataset, desc="Processing episodes"):
        # Each episode has a 'steps' feature which is a Dataset of steps
        for step in episode['steps'].as_numpy_iterator():
            all_actions.append(step['action'])
            total_steps += 1

    if not all_actions:
        print("Error: No actions found in the combined dataset.")
        return

    # Stack all actions into a single numpy array
    all_actions = np.stack(all_actions)
    print(f"\nCollected {all_actions.shape[0]} actions with dimension {all_actions.shape[1]}")
    print(f"Total number of steps found: {total_steps}")

    # --- Calculate Percentile Bins ---
    action_dim = all_actions.shape[1]
    action_bins = []

    print(f"Calculating {num_bins} percentile bins for each of the {action_dim} action dimensions...")
    for i in range(action_dim):
        dim_actions = all_actions[:, i]
        # We need num_bins + 1 points to define num_bins intervals.
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(dim_actions, percentiles)
        
        # Ensure the bins are monotonically increasing to handle cases where
        # percentiles might be identical (e.g., if many actions are zero).
        for j in range(1, len(bins)):
            if bins[j] <= bins[j-1]:
                bins[j] = bins[j-1] + 1e-6  # Add a small epsilon
        action_bins.append(bins)

    action_bins = np.array(action_bins)

    # Save the bins to a file for later use
    np.save(output_path, action_bins)
    print(f"\nAction bin boundaries calculated with shape: {action_bins.shape}")
    print(f"Saved action bins to '{output_path}'")

    # Save the dataset info (total steps and per-dataset steps) to a JSON file
    dataset_info = {
        "total_steps": total_steps,
        "dataset_steps": dataset_steps,
        "dataset_paths": [os.path.basename(path) for path in dataset_paths]
    }
    with open(info_output_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    print(f"Saved dataset info to '{info_output_path}'")
    print(f"Dataset breakdown:")
    for i, (path, steps) in enumerate(zip(dataset_paths, dataset_steps)):
        print(f"  {os.path.basename(path)}: {steps} steps")

def main():
    parser = argparse.ArgumentParser(description="Calculate and save action bins from multiple RLDS datasets.")
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="/home/timely/lklein/vla/modified_libero_rlds",
        help="Base path containing the libero_* dataset directories."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=256,
        help="Number of discrete bins for each action dimension."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="action_bins.npy",
        help="Path to save the output .npy file."
    )
    parser.add_argument(
        "--info_output_path",
        type=str,
        default="action_bins_info.json",
        help="Path to save the dataset info JSON file."
    )
    args = parser.parse_args()

    calculate_and_save_action_bins(args.dataset_base_path, args.num_bins, args.output_path, args.info_output_path)

if __name__ == "__main__":
    main()
