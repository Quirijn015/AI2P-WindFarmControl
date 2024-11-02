import os

import numpy as np
import torch
import json

from architecture.LSTMUNet.lstmu_net import LSTMUNet
from experiments.LSTMUNet_experiments.lstmunet_dataset import create_data_loaders, get_dataset
from utils.preprocessing import resize_windspeed
from utils.visualization import plot_prediction_vs_real, animate_prediction_vs_real

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def create_transform(scale):
    def resize_scalars(windspeed_scalars):
        return [resize_windspeed(scalar, scale) for scalar in windspeed_scalars]
    return resize_scalars


def make_model_predictions(model, inputs, length):
    if length <= 0:
        return model(inputs)
    outputs = model(inputs)
    print(f"Ïntermediate outputs: {outputs.shape}")
    next_outputs = make_model_predictions(model, outputs, length - outputs.shape[1])
    return torch.cat((outputs.squeeze(), next_outputs.squeeze()), dim=0)


def get_model_targets(dataset, index, length):
    if length <= 0:
        _, targets = dataset[index]
        return targets
    _, targets = dataset[index]
    sequence_length = targets.shape[0]
    next_targets = get_model_targets(dataset, index + sequence_length, length - sequence_length)
    return torch.cat((targets, next_targets), dim=0)


def plot(animate: bool):

    latest = max(os.listdir("LSTMUNet_parameters"))

    with open(f"LSTMUNet_parameters/config.json", "r") as f:
        config = json.load(f)

    case = 1
    sequence_length = config["sequence_length"]
    batch_size = config["batch_size"]
    scale = config["scale"]

    transform = create_transform(scale)
    dataset = get_dataset(config["dataset_dirs"], sequence_length, transform)

    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size)
    model = LSTMUNet(sequence_length).to(device)

    max_epoch = max(os.listdir(f'LSTMUNet_parameters/{latest}'))
    model.load_state_dict(torch.load(f"LSTMUNet_parameters/{latest}"))
    model.eval()

    print(f"LSTMUNet_parameters/{latest}/{max_epoch}")


    with torch.no_grad():
        if animate:
            animation_length = 50
            start = np.random.randint(0, min(1, len(dataset) - max(sequence_length, animation_length)))
            inputs, _ = dataset[start]
            outputs = make_model_predictions(model, inputs[None, :, :, :], animation_length).squeeze()
            print(f"Outputs.shape: {outputs.shape}")
            targets = get_model_targets(dataset, start, animation_length).squeeze()
            print(f"Targets.shape: {targets.shape}")

            def animate_callback(i):
                return outputs[i], targets[i]

            animate_prediction_vs_real(animate_callback, animation_length, f"results/{latest}")

        else:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                plot_prediction_vs_real(output[0, 45, :, :].cpu(), targets[0, 45, :, :].cpu(), case)

if __name__ == '__main__':
    plot(False)
    # plot(True)
