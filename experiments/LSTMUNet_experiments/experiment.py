import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from windspeedLSTM.windspeedLSTM import (WindspeedLSTM)
from experiments.futureWindspeedLSTM.WindspeedMapDataset import create_data_loaders, get_dataset
from utils.preprocessing import resize_windspeed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_config(case):
    return {
        "case": case,
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


def create_transform(scale):
    def resize_scalars(windspeed_scalars):
        return [resize_windspeed(scalar, scale) for scalar in windspeed_scalars]

    return resize_scalars


def run():
    case = 123
    config = load_config(case)

    sequence_length = config["sequence_length"]
    batch_size = config["batch_size"]
    scale = config["scale"]
    transform = create_transform(scale)

    dataset = get_dataset(config["dataset_dirs"], sequence_length, transform)

    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size,Random=False)
    model = WindspeedLSTM(sequence_length).to(device)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    output_folder = create_output_folder(case)

    save_config(output_folder, config)

    train(model, train_loader, val_loader, output_folder)


def train(model, train_loader, val_loader, output_folder):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    num_epochs = 50
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_losses = []
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_losses = []
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                val_losses.append(val_loss.item())
            model.train()

        learning_rate = optimizer.param_groups[0]['lr']
        avg_val_loss = np.mean(val_losses)
        print(
            f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {np.mean(train_losses)}, validation loss: {avg_val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/model/model_01_{epoch}.pt")

        # Check early stopping criterion
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 8:
            print(f'Early stopping at epoch {epoch}')
            break


def create_output_folder(case_nr):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{case_nr}"
    os.makedirs(f"{output_folder}/model")
    return output_folder


def save_config(output_folder, config):
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    run()

