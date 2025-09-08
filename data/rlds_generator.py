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
    def __init__(self, rlds_paths: list[str], tokenizer: AutoTokenizer, action_bins_path: str, action_bins_info_path: str, action_token_begin_id: int, split: str = 'train', val_ratio: float = 0.025):
        super().__init__()
        self.rlds_paths = rlds_paths
        self.tokenizer = tokenizer
        self.action_token_begin_id = action_token_begin_id
        self.split = split
        
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        self.val_every_n = int(1 / val_ratio)

        # 1. Load the pre-computed action bin boundaries
        print(f"Generator ({self.split}): Loading action bins from: {action_bins_path}")
        self.action_bins = np.load(action_bins_path)
        self.num_bins = self.action_bins.shape[1] - 1
        self.action_dim = self.action_bins.shape[0]
        print(f"Generator ({self.split}): Loaded action bins for {self.action_dim} dimensions with {self.num_bins} bins each.")
        print(f"Generator ({self.split}): Using action token start ID: {self.action_token_begin_id}")
        print(f"Generator ({self.split}): Yielding every {self.val_every_n}-th sample for validation.")

        # 2. Load pre-computed dataset info (total steps)
        print(f"Generator ({self.split}): Loading dataset info from: {action_bins_info_path}")
        with open(action_bins_info_path, 'r') as f:
            dataset_info = json.load(f)
        total_steps = dataset_info['total_steps']

        # 3. Calculate the length of this split
        num_val_samples = total_steps // self.val_every_n
        num_train_samples = total_steps - num_val_samples

        if self.split == 'train':
            self._len = num_train_samples
        else:
            self._len = num_val_samples
        
        print(f"Generator ({self.split}): Pre-calculated length is {self._len} samples.")

    def _create_dataset_stream(self):
        # This function is now separate to be called by both __init__ and __iter__
        datasets = []
        for path in self.rlds_paths:
            builder = tfds.builder_from_directory(builder_dir=path)
            ds = builder.as_dataset(split='train')
            datasets.append(ds)
        
        interleaved_dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=True)
        return interleaved_dataset

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
        interleaved_dataset = self._create_dataset_stream()
        
        # The main data processing loop
        step_idx = 0
        for episode in interleaved_dataset:
            for step in episode['steps'].as_numpy_iterator():
                # Determine if this step belongs to the current split
                is_val_sample = (step_idx % self.val_every_n == 0)
                step_idx += 1

                if self.split == 'train' and is_val_sample:
                    continue
                if self.split == 'validation' and not is_val_sample:
                    continue

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