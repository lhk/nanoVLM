"""
rlds_data_generator.py

Defines a PyTorch IterableDataset that serves as a simple generator.
It reads from an RLDS dataset, discretizes actions, and yields the data
in a format compatible with the VQADataset class.
"""
import torch
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from PIL import Image
import glob
import os

# Disable noisy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')


import json


class RLDSDataGenerator(IterableDataset):
    """
    An IterableDataset that acts as a generator, streaming data from RLDS.

    This class reads raw data from one or more RLDS datasets, performs action
    discretization, and formats the output into a simple dictionary that can be
    consumed by a more complex downstream dataset, like `VQADataset`.
    """
    def __init__(self, rlds_paths: list[str], tokenizer: AutoTokenizer, action_bins_path: str, action_bins_info_path: str, action_token_begin_id: int, early_stop_after_n_samples: int = None):
        super().__init__()
        self.rlds_paths = rlds_paths
        self.tokenizer = tokenizer
        self.action_token_begin_id = action_token_begin_id
        self.early_stop_after_n_samples = early_stop_after_n_samples

        # 1. Load the pre-computed action bin boundaries
        print(f"Loading action bins from: {action_bins_path}")
        self.action_bins = np.load(action_bins_path)
        self.num_bins = self.action_bins.shape[1] - 1
        self.action_dim = self.action_bins.shape[0]
        print(f"Loaded action bins for {self.action_dim} dimensions with {self.num_bins} bins each.")
        print(f"Using action token start ID: {self.action_token_begin_id}")

        # 2. Load pre-computed dataset info and calculate length for the specified datasets
        print(f"Loading dataset info from: {action_bins_info_path}")
        with open(action_bins_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        # Calculate length based on which datasets this generator will use
        dataset_steps = dataset_info['dataset_steps']
        dataset_names = dataset_info.get('dataset_paths', [])
        
        # Calculate total length for the datasets this generator will process
        total_length = 0
        for path in self.rlds_paths:
            dataset_name = os.path.basename(path)
            if dataset_name in dataset_names:
                idx = dataset_names.index(dataset_name)
                assert idx != -1, f"Dataset {dataset_name} not found in dataset info."
                if idx < len(dataset_steps):
                    total_length += dataset_steps[idx]


        self._len = min(self.early_stop_after_n_samples if self.early_stop_after_n_samples is not None else total_length, total_length)
        print(f"Calculated length for {len(self.rlds_paths)} datasets: {self._len} samples" + 
              (f" (early stop after {self.early_stop_after_n_samples} samples)" if self.early_stop_after_n_samples else ""))

    def _create_dataset_stream(self):
        # This function is now separate to be called by both __init__ and __iter__
        datasets = []
        for path in self.rlds_paths:
            builder = tfds.builder_from_directory(builder_dir=path)
            ds = builder.as_dataset(split='train')
            datasets.append(ds)
        
        concatenated_dataset = datasets[0]
        for ds in datasets[1:]:
            concatenated_dataset = concatenated_dataset.concatenate(ds)

        return concatenated_dataset.shuffle(buffer_size=100)
    
    def __len__(self):
        return self._len
        
    def _discretize_action(self, continuous_action: np.ndarray) -> np.ndarray:
        """Converts a continuous action vector to discrete bin indices."""
        min_vals = self.action_bins[:, 0]
        max_vals = self.action_bins[:, -1]
        continuous_action = np.clip(continuous_action, min_vals, max_vals)
        
        binned_action = []
        for i in range(self.action_dim):
            bin_index = np.searchsorted(self.action_bins[i], continuous_action[i], side='right') - 1
            binned_action.append(np.clip(bin_index, 0, self.num_bins - 1))
        return np.array(binned_action)

    def _get_action_token_ids(self, binned_action: np.ndarray) -> list[int]:
        """Converts discrete bin indices to their corresponding vocabulary token IDs."""
        token_ids = []
        for i in range(self.action_dim):
            token_id = self.action_token_begin_id + (i * self.num_bins) + binned_action[i]
            token_ids.append(token_id)
        return token_ids

    def __iter__(self):
        # Re-create the dataset stream for each epoch
        iter_dataset = self._create_dataset_stream()
        
        # Track sample count for early stopping
        sample_count = 0
        
        # The main data processing loop
        for episode in iter_dataset:
            for step in episode['steps'].as_numpy_iterator():
                # Check for early stopping
                if self.early_stop_after_n_samples is not None and sample_count >= self.early_stop_after_n_samples:
                    return  # This ends the iterator

                # a. Get raw data
                instruction = step['language_instruction'].decode('utf-8')
                image = Image.fromarray(step['observation']['image'])
                continuous_action = step['action']

                # b. Discretize action and convert to a string of tokens
                binned_action = self._discretize_action(continuous_action)
                action_token_ids = self._get_action_token_ids(binned_action)
                action_string = self.tokenizer.decode(action_token_ids)

                # c. Yield in the format expected by VQADataset
                yield {
                    'images': [image],
                    'texts': [{'user': instruction, 'assistant': action_string}]
                }
                
                sample_count += 1

if __name__ == '__main__':
    # This is a test harness to demonstrate how to use the generator.
    print("--- Running RLDSDataGenerator Test ---")

    # 1. Define paths and parameters
    base_path = "/home/timely/lklein/vla/modified_libero_rlds"
    dataset_paths = glob.glob(f"{base_path}/libero_*_no_noops/1.0.0")
    action_bins_path = "action_bins.npy"
    
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found at '{base_path}'. Please check the path.")
    
    print(f"Found {len(dataset_paths)} datasets.")

    # 2. Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    # In a real run, special tokens would be added by get_tokenizer in train.py
    # We manually add one here to simulate the post-addition vocab size.
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})


    # 3. Calculate the action token start ID *after* all special tokens are added
    ACTION_BINS_FOR_TEST = np.random.rand(7, 257) # Dummy action bins for test
    num_action_tokens = ACTION_BINS_FOR_TEST.shape[0] * (ACTION_BINS_FOR_TEST.shape[1] - 1)
    action_token_begin_id = len(tokenizer) - num_action_tokens
    print(f"Test: Calculated action_token_begin_id: {action_token_begin_id}")


    # 4. Instantiate the generator for both splits
    train_generator = RLDSDataGenerator(
        rlds_paths=dataset_paths,
        tokenizer=tokenizer,
        action_bins_path=action_bins_path,
        action_token_begin_id=action_token_begin_id,
        split='train'
    )
    val_generator = RLDSDataGenerator(
        rlds_paths=dataset_paths,
        tokenizer=tokenizer,
        action_bins_path=action_bins_path,
        action_token_begin_id=action_token_begin_id,
        split='validation'
    )

    # 5. Load and inspect a single sample from each split
    print("\n--- Loading one sample from the TRAIN generator ---")
    try:
        # NOTE: This test requires a valid `action_bins.npy` to exist.
        # Run `rlds_action_discretization.py` first.
        sample = next(iter(train_generator))
        print("Successfully loaded one train sample.")
        print("Sample 'texts' content:", sample['texts'])

    except StopIteration:
        print("Could not retrieve a sample from the train generator.")
    except FileNotFoundError:
        print("\nERROR: `action_bins.npy` not found. Please run `rlds_action_discretization.py` before this test.")
    except Exception as e:
        print(f"An error occurred while loading a train sample: {e}")

    print("\n--- Loading one sample from the VALIDATION generator ---")
    try:
        sample = next(iter(val_generator))
        print("Successfully loaded one validation sample.")
        print("Sample 'texts' content:", sample['texts'])

    except StopIteration:
        print("Could not retrieve a sample from the validation generator.")
    except FileNotFoundError:
        print("\nERROR: `action_bins.npy` not found. Please run `rlds_action_discretization.py` before this test.")
    except Exception as e:
        print(f"An error occurred while loading a validation sample: {e}")

    print("\n--- Test Complete ---")