"""Loading functionality for toy MPC benchmark (PyTorch version, GPU ready)."""

import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ============================================================
#   DATASET
# ============================================================

class ToyMPCDataset(Dataset):
    """Dataset for toy MPC benchmark."""

    def __init__(self, data: dict, const: dict):
        """
        Args:
            data (dict): Dictionary containing the dataset.
            const (dict): Dictionary containing constant problem ingredients.
        """
        self.x0sets = torch.tensor(data["x0sets"], dtype=torch.float32)
        self.objectives = torch.tensor(data["objectives"], dtype=torch.float32)
        self.Ystar = torch.tensor(data["Ystar"], dtype=torch.float32)
        self.const = (
            torch.tensor(const["As"], dtype=torch.float32),
            torch.tensor(const["lbxs"], dtype=torch.float32),
            torch.tensor(const["ubxs"], dtype=torch.float32),
            torch.tensor(const["lbus"], dtype=torch.float32),
            torch.tensor(const["ubus"], dtype=torch.float32),
            torch.tensor(const["xhat"], dtype=torch.float32),
            const["alpha"],
            const["T"],
            const["base_dim"],
        )

    def __len__(self) -> int:
        return self.x0sets.shape[0]

    def __getitem__(self, idx: int):
        return self.x0sets[idx], self.objectives[idx]


# ============================================================
#   DATALOADERS
# ============================================================

def create_dataloaders(
    dataset: ToyMPCDataset,
    batch_size: int = 2048,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for training, validation, and testing."""
    size = len(dataset)
    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
#   LOAD DATA
# ============================================================

def load_data(
    filepath: str,
    batch_size: int = 2048,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, float, int, int,
    torch.Tensor,
    DataLoader, DataLoader, DataLoader,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Load toy MPC problem data as PyTorch tensors and create dataloaders."""
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filepath)
    all_data = np.load(dataset_path, allow_pickle=True)
    dataset = ToyMPCDataset(all_data, all_data)

    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
    )

    As, lbxs, ubxs, lbus, ubus, xhat, alpha, T, base_dim = dataset.const
    X = dataset.x0sets
    dimx = lbxs.shape[1]

    def batched_objective(batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the quadratic objective for a batch in a fully vectorized way.
        
        batch: (batch_size, dim_prediction)
        Returns: (batch_size,) objective values
        """
        device = batch.device
        state_pred = batch[:, :dimx]  # (batch_size, dimx)
        control_pred = batch[:, dimx:]  # (batch_size, dimu)

        # Tile xhat across batch
        xhat_tile = xhat[:, 0].repeat(T + 1).to(device)  # (dimx,)
        xhat_tile = xhat_tile.unsqueeze(0).expand(batch.shape[0], -1)  # (batch_size, dimx)

        alpha_tensor = torch.tensor(alpha, device=device)

        obj_state = torch.sum((state_pred - xhat_tile) ** 2, dim=1)  # (batch_size,)
        obj_control = alpha_tensor * torch.sum(control_pred ** 2, dim=1)  # (batch_size,)

        return obj_state + obj_control

    return (
        As, lbxs, ubxs, lbus, ubus,
        xhat, alpha, T, base_dim,
        X,
        train_loader, valid_loader, test_loader,
        batched_objective,
    )