import os
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset

class ThumanDataset(Dataset):
    def __init__(self, split='train', data_root='data/Thuman', label_file=None, test_ratio=0.2, voxel_size=0.02, voxel_max=None, transform=None, shuffle_index=False, loop=1):
      
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.transform = transform
        self.voxel_max = voxel_max
        self.shuffle_index = shuffle_index
        self.loop = loop

        # List all .ply files in the dataset directory
        data_list = sorted([file for file in os.listdir(data_root) if file.endswith('.ply')])
        data_list = [os.path.join(data_root, file) for file in data_list]

        # Split dataset into train and test sets
        split_idx = int(len(data_list) * (1 - test_ratio))
        if split == 'train':
            self.data_list = data_list[:split_idx]
        else:
            self.data_list = data_list[split_idx:]

        # Load labels if label_file is provided
        self.labels = None
        if label_file:
            self.labels = self._load_labels(label_file)

        self.data_idx = np.arange(len(self.data_list))
        print(f"Totally {len(self.data_idx)} samples in {split} set.")

    def _load_labels(self, label_file):
        """
        Load labels from an external file.

        Args:
            label_file (str): Path to the label file.

        Returns:
            dict: Mapping from file name to labels.
        """
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                label = int(parts[1])  # Assuming labels are integers
                labels[filename] = label
        return labels

    def __getitem__(self, idx):
        """
        Retrieve a single data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            coord (np.ndarray): Array of point coordinates (xyz).
            feat (np.ndarray): Array of point features (rgb).
            label (np.ndarray): Array of point labels.
        """
        # Get the file path for the current index
        file_path = self.data_list[idx % len(self.data_list)]

        # Load the .ply file
        ply_data = PlyData.read(file_path)
        data = np.vstack([ply_data['vertex'][axis] for axis in ['x', 'y', 'z', 'red', 'green', 'blue']]).T

        # Check for labels in the file
        if 'label' in ply_data['vertex'].dtype.names:
            label = np.array(ply_data['vertex']['label'])
        elif self.labels is not None:
            # If labels are provided in a separate file, map them
            filename = os.path.basename(file_path)
            label_value = self.labels[filename]
            label = np.full(data.shape[0], label_value, dtype=np.int32)
        else:
            raise ValueError(f"Labels not found in file {file_path} and no external label file provided.")

        # Split into coordinates, features, and labels
        coord, feat = data[:, :3], data[:, 3:6]

        # Apply optional data preparation or augmentation
        if self.transform:
            coord, feat, label = self.transform(coord, feat, label)

        # Shuffle point indices if specified
        if self.shuffle_index:
            indices = np.random.permutation(len(coord))
            coord, feat, label = coord[indices], feat[indices], label[indices]

        return coord, feat, label

    def __len__(self):
        """
        Return the total number of samples.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.data_idx) * self.loop
