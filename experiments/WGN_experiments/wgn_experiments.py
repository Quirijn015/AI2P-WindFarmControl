
import json
import os
import torch

import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from architecture.WGN.deconv import FCDeConvNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WGNDataset(Dataset):
    def __init__(self, data_folder, max_angle):
        self.data_folder = data_folder
        self.max_angle = max_angle
        # Get list of all files
        self.dataset = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the data, e.g., an image or tensor from file
        grid_path = self.dataset[idx]
        graph_data = torch.load(grid_path, weights_only=False)

        return graph_data

def custom_collate_fn(batch):
    # Each batch consists of a list of sequences (each a list of graphs)
    return [Data.from_data_list(seq) for seq in batch]


def create_data_loaders(dataset, batch_size, seq_length):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    collate = custom_collate_fn if seq_length > 1 else None
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_loss(batch, criterion, model):
    # Logic to handle different model types
    x, pos, edge_attr, glob, target = batch.x, batch.pos, batch.edge_attr.float(), batch.global_feats.float(), batch.y
    # Concatenate features for non-GNN models
    if isinstance(model, FCDeConvNet):
        x_cat = torch.cat([x.flatten(), pos.flatten(), edge_attr.flatten(), glob.flatten()], dim=-1).float()
        pred = model(x_cat)
    else:
        nf = torch.cat((x, pos), dim=-1).float()
        pred = model(batch, nf, edge_attr, glob)
    loss = criterion(pred, target.reshape((pred.size(0), -1, 128, 128)))
    return loss


def train_epoch(train_loader, model, criterion, optimizer, scheduler):
    train_losses = []
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss = compute_loss(batch, criterion, model)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_epoch(val_loader, model, criterion):
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            batch = batch.to(device)
            val_loss = compute_loss(batch, criterion, model)
            val_losses.append(val_loss.item())
    model.train()
    return np.mean(val_losses)


def train(model, train_params, train_loader, val_loader, output_folder):
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss = eval_epoch(val_loader, model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/WGN_{epoch}.pt")
        if epoch==num_epochs:
            np.save("train_data/train_loss", train_loss_list)
            np.save("train_data/val_loss", val_loss_list)
        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{output_folder}/WGN_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            print(f'Early stopping at epoch {epoch}')
            np.save("train_data/train_loss", train_loss_list)
            np.save("train_data/val_loss", val_loss_list)
            break

def get_wgn_config():
    return {
        'edge_in_dim': 2,
        'node_in_dim': 4,
        'global_in_dim': 2,
        'n_pign_layers': 3,
        'edge_hidden_dim': 50,
        'node_hidden_dim': 50,
        'global_hidden_dim': 50,
        'num_nodes': 10,
        'residual': True,
        'input_norm': True,
        'pign_mlp_params': {
            'num_neurons': [256, 128],
            'hidden_act': 'ReLU',
            'out_act': 'ReLU'
        },
        'mlp_params': {
            'num_neurons': [128, 128, 64],
        },
        'deconv_params':{
            'layer_channels': [64, 128, 256, 1],
        }
}


def get_config(case_nr, wake_steering, max_angle, use_graph, seq_length, batch_size, output_size, direct_lstm=False, num_epochs=200, early_stop_after=10):
    return get_wgn_config(), {
        'case_nr': case_nr,
        'wake_steering': wake_steering,
        'max_angle': max_angle,
        'num_epochs': num_epochs,
        'use_graph': use_graph,
        'early_stop_after': early_stop_after,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'direct_lstm': direct_lstm,
        'output_size': output_size,
    }


