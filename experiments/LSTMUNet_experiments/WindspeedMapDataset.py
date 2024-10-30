import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from torch.utils.data import Subset
import random

# class TemporalDataset(Dataset):
#     def __init__(self, root, seq_length, preload=True):
#
#         self.root = root
#         self.seq_length = seq_length
#         self.preload = preload
#         self.len = len([name for name in os.listdir(self.root)]) - 2 * self.seq_length
#
#         if preload:
#             self.data = [torch.tensor(np.load(f"{self.root}/Windspeed_map_scalars_{30005 + (start + i) * 5}.npy")) for start in range(self.len) for i in range(self.seq_length)]
#
#     def _get_sequence(self, start):
#         if self.preload:
#             return [self.data[start + i] for i in range(self.seq_length)]
#         else:
#             return [torch.tensor(np.load(f"{self.root}/Windspeed_map_scalars_{30005 + (start + i) * 5}.npy")) for i in range(self.seq_length)]
#
#     def __len__(self):
#         return self.len
#
#     def  __getitem__(self, idx):
#         return self._get_sequence(idx), self._get_sequence(idx + self.seq_length)
class WindspeedMapDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        # Update length to allow for sliding windows
        self.len = len([name for name in os.listdir(root_dir) if name != "README.md"]) - 2*sequence_length

    def __len__(self):
        return self.len

    def _get_sequence(self, start, transform):
        # Use a sliding window with a step of 1
        scalars = [np.load(f'{self.root_dir}/Windspeed_map_scalars_{30005 + (start + i) * 5}.npy')
                   for i in range(self.sequence_length)]
        if transform:
            scalars = transform(scalars)
        scalars = torch.tensor(np.array(scalars), dtype=torch.float32)
        return scalars

    def __getitem__(self, idx):
        # Keep the current sequence, and set the target as the next sequence starting one timestep later
        scalars = self._get_sequence(idx, self.transform)
        target_scalars = self._get_sequence(idx + self.sequence_length, self.target_transform)
        return scalars, target_scalars

# class WindspeedMapDataset(Dataset):
#     def __init__(self, root_dir, sequence_length, transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.sequence_length = sequence_length
#         self.len = len([name for name in os.listdir(root_dir) if name != "README.md"]) // sequence_length - 2
#
#     def __len__(self):
#         return self.len
#
#
#     def _get_sequence(self, start, transform):
#         scalars = [np.load(f'{self.root_dir}/Windspeed_map_scalars_{30005 + (start * self.sequence_length + i) * 5}.npy') for i in range(self.sequence_length)]
#         if transform:
#             scalars = transform(scalars)
#         scalars = torch.tensor(np.array(scalars), dtype=torch.float32)
#         return scalars
#
#     def __getitem__(self, idx):
#         scalars = self._get_sequence(idx, self.transform)
#         target_scalars = self._get_sequence(idx + 1, self.target_transform)
#         return scalars, target_scalars

def get_dataset(dataset_dirs, sequence_length, transform):
    datasets = []
    for path in dataset_dirs:
        dataset = WindspeedMapDataset(path, sequence_length)
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    print(f"Loaded datasets, {len(dataset)} samples")
    return dataset

def create_data_loaders(dataset, batch_size,Random):
    total_len = len(dataset)

    if Random is True:
        torch.manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
    else:
        random.seed(42)
        data_ind = range(0, total_len)  # Full range of data indices

        # Sample 60 start indices for sequences
        sequence_starts = random.sample(range(0, total_len - 50), 20)  # Ensure sequences fit within data range

        # Expand each start index to a sequence of 50 elements
        test_val_ind = [list(range(i, i + 50)) for i in sequence_starts]  # 60 sequences of 50 indices each

        # Flatten the list of lists for easier manipulation
        test_val_ind = [idx for seq in test_val_ind for idx in seq]

        # Split into test and validation sets
        split_index = len(test_val_ind) // 2
        test_indices = test_val_ind[split_index:]
        val_indices = test_val_ind[:split_index]

        # Remove test and validation indices from the training set
        train_indices = list(set(data_ind) - set(test_indices) - set(val_indices))
        print("training set length",  len(train_indices))
        print("validation set length", len(val_indices))
        print("test set length", len(test_indices))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    torch.save(test_dataset, 'test_dataset')

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    config = {
        "case": 123,
        "dataset_dirs": [
            f"../../Data/target_data_resized/Case_01/postProcessing_BL",
            f"../../Data/target_data_resized/Case_01/postProcessing_LuT2deg_internal",
            f"../../Data/target_data_resized/Case_02/postProcessing_BL",
            f"../../Data/target_data_resized/Case_02/postProcessing_LuT2deg_internal",
            # f"../../Data/target_data_resized/Case_03/postProcessing_BL",
            # f"../../Data/target_data_resized/Case_03/postProcessing_LuT2deg_internal",
        ],
        "sequence_length": 50,
        "batch_size": 1,
        "scale": (128, 128)
    }
    sequence_length = config["sequence_length"]
    batch_size = config["batch_size"]
    scale = config["scale"]
    transform = None

    dataset = get_dataset(config["dataset_dirs"], sequence_length, transform)

    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size, Random=False)
    print(train_loader)
