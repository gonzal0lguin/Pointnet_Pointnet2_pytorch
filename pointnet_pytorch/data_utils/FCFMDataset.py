import os
import numpy as np
from torch.utils.data import Dataset
import pickle
from random import shuffle
from tqdm import tqdm

class FCFMDataset(Dataset):
    def __init__(self, 
                 root, 
                 num_classes=2,
                 num_point=2048*3, 
                 norm_xyz=True,
                 norm_intensity=True, 
                 dn_cloud=False,
                 preload=False,
                 test_paths=None,
                 percent=0.8):
        """
        Args:
            data_paths: list of npy files. Each file is expected to have shape (N, 5) -> [x, y, z, intensity, label]
            num_point: number of points per sample (e.g., 4096)
            transform: optional data augmentation
        """
        self.data_root = root
        self.preload = preload
        self.num_classes = num_classes
        self.num_point = num_point
        self.norm_xyz  = norm_xyz
        self.norm_intensity = norm_intensity 
        self.xyz_norm_value = 50.0

        self.cloud_name = 'lidar' if not dn_cloud else 'lidar_dn'


        if test_paths is None:
            data_paths = os.listdir(root)  # List of (N, 5) arrays
            # shuffle the data paths for randomness
            shuffle(data_paths)
        
            split_index = int(len(data_paths) * percent)
            self.data_paths = data_paths[:split_index]
            self.test_data_paths = data_paths[split_index:]

        else:
            self.data_paths = test_paths  # List of (N, 5) arrays

        if preload:
            self.preload_data_and_labelweights()


    def compute_class_weights(self):
        labelweights = np.zeros(13)        
        if hasattr(self, 'data_paths'):
            label_counts = {}
            labelweights = np.zeros(self.num_classes)

            for path in self.data_paths:
                data = self._read_pkl(os.path.join(self.data_root, path))
                labels = data[self.cloud_name][:, 4]
                unique_labels, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += count
            total_count = sum(label_counts.values())
            class_weights = {label: total_count / count for label, count in label_counts.items()}
            return class_weights
        else:
            raise AttributeError("Data paths are not available. Ensure 'train' is set to True during initialization.")
        

    def preload_data_and_labelweights(self):
        labelweights = np.zeros(self.num_classes)
        self.points = []
        self.labels = []
        self.xyz_max, self.xyz_min = 0, 0

        for file in tqdm(self.data_paths, total=len(self.data_paths), desc="Preloading data..."):
            filepath = os.path.join(self.data_root, file)
            if self.cloud_name == 'lidar_dn':
                cloud_data = np.stack(self._read_pkl(filepath)[self.cloud_name]).reshape(-1, 5)  # xyzil, N*5
            else:
                raw_lidar = self._read_pkl(filepath)[self.cloud_name]
                cloud_data = self._process_lidar(raw_lidar, max_points=self.num_point).reshape(-1, 5) # todo hacer que calcen las dims xdxd

            points, labels = cloud_data[:, 0:4], cloud_data[:, 4].astype(int)  # xyzi, N*4; l, N
            tmp, _ = np.histogram(labels, range(self.num_classes + 1))
            labelweights += tmp
            self.points.append(points), self.labels.append(labels)
            
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.xyz_min = np.minimum(self.xyz_min, coord_min)
            self.xyz_max = np.maximum(self.xyz_max, coord_max)
        
        print("XYZ min:", self.xyz_min, "XYZ max:", self.xyz_max)

        # Normalize and re-weight
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)  # get class distribution
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # soften the imbalance
        print("Class weights:", self.labelweights)


    def get_test_data_paths(self):
        if hasattr(self, 'test_data_paths'):
            return self.test_data_paths
        else:
            raise AttributeError("Test data paths are not available. Ensure 'train' is set to True during initialization.")

    def _read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def _process_lidar(batched_pts, voxel_size=0.08, max_points=5120):
        process_lidar = []
        for points in batched_pts:
            coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
            _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)

            # Sum xyz and intensity by voxel
            # xyz_intensity = np.concatenate([points[:, :3], points[:, 3]], axis=1)
            sums = np.zeros((counts.shape[0], 4), dtype=np.float32)
            np.add.at(sums, inv, points[:, :4])

            # Divide by counts to get mean per voxel
            means = sums / counts[:, None]

            N = means.shape[0]
            if N > max_points:
                indices = np.random.choice(N, max_points, replace=False)
                means = means[indices]

                if points.shape[1] == 5:
                    labels = points[indices, 4]
                    means = np.concatenate([means, labels])

            elif N < max_points:
                pad = np.zeros((max_points - N, 4), dtype=np.float32)
                means = np.concatenate((means, pad), axis=0)
            
            process_lidar.append(means)

        return np.array(process_lidar)


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        if self.preload:
            xyz = self.points[idx][:, :3]
            intensity = self.points[idx][:, 3]
            labels = self.labels[idx]

        else:
            if self.cloud_name == 'lidar_dn':
                data = self._read_pkl(os.path.join(self.data_root, self.data_paths[idx]))
                cloud = np.stack(data[self.cloud_name]).reshape(-1, 5)
            else:
                raw_lidar = self._read_pkl(os.path.join(self.data_root, self.data_paths[idx]))[self.cloud_name]
                cloud = self._process_lidar(raw_lidar, max_points=self.num_point).reshape(-1, 5)

            xyz = cloud[:, :3]
            intensity = cloud[:, 3]
            labels = cloud[:, 4].astype(np.int64)

        N = xyz.shape[0]
        if N >= self.num_point:
            idxs = np.random.choice(N, self.num_point, replace=False)
        else:
            idxs = np.random.choice(N, self.num_point, replace=True)

        selected_xyz = xyz[idxs]
        selected_intensity = intensity[idxs]
        selected_labels = labels[idxs]

        # Normalize XYZ (optional)
        if self.norm_xyz:
            selected_xyz = selected_xyz / self.xyz_norm_value
            # selected_xyz = selected_xyz - selected_xyz.mean(0, keepdims=True)
        if self.norm_intensity:
            selected_intensity = (selected_intensity - np.mean(selected_intensity)) / (np.std(selected_intensity) + 1e-6)

        # Combine features: [x, y, z, intensity]
        points = np.concatenate([selected_xyz, selected_intensity.reshape(-1, 1)], axis=1)  # shape (num_point, 4)

        return points.astype(np.float32), selected_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pointnet_pytorch.models.pointnet2_sem_seg import get_model
    import torch

    a = FCFMDataset(root='/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120/cc/', 
                    num_point=5120//2*3, 
                    dn_cloud=True, 
                    preload=True)
    
    test_paths = a.get_test_data_paths()  # Ensure test paths are set
    print("Number of training samples:", len(a))

    b = FCFMDataset(root='/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120/cc/',
                    num_point=5120//2*3,
                    dn_cloud=True,
                    test_paths=test_paths)
    
    print("Number of test samples:", len(b))

    print("Sample train data shape:", a[0][0].shape, "Sample label shape:", a[0][1].shape)
    print("Sample test data shape:", b[0][0].shape, "Sample label shape:", b[0][1].shape)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = get_model(5120//2*3, 2).to(device)
    # val = torch.tensor(a[0][0], device=device, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # val = val.permute(0, 2, 1)  # Change to (batch_size, num_features, num_points)
    # target = torch.tensor(a[0][1], device=device, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # target = target.view(-1, 1)[:, 0]
    # # print(val.shape)
    
    # log_probs, _ = model(val)
    # seg_pred = log_probs.contiguous().view(-1, 2) # (num_points, num_classes)
    
    # print(seg_pred, seg_pred.shape)
    # print(target, target.shape)

    # loss = torch.nn.functional.nll_loss(seg_pred, target)
    # print("Loss:", loss.item())
    
    # # print(len(a))
    # # print(a[0])
    # labels = a[0][1]
    # print(log_probs.shape, labels.shape)
    # print(log_probs[:, 0, :], labels[0])
    # print(log_probs[:, 0, labels[0]])
    # plt.show()