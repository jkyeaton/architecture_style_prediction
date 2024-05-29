#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Import datasets
sys.path.append("..")
# Import precision and accuracy metrics from package
from data.dumitrux_dataset import DumitruxDataset  # noqa: E402
from utils.eval_metrics import (  # noqa: E402
    precision_per_class,
    recall_per_class,
    total_accuracy,
)


def main(config, out_dir, transfer, checkpoint):
    """
    Executes the main training and evaluation workflow.

    This function orchestrates the model training and evaluation process using
    specified configurations. It involves data loading, model initialization,
    training, evaluation, and result visualization.

    Args:
        config (dict): Configuration parameters for the model and training process.
        out_dir (str): Directory where output files (like plots) will be saved.
        transfer (bool): Flag indicating whether to use transfer learning.
        checkpoint (str): Path to a pre-trained model checkpoint.

    Returns:
        None
    """
    # Unpack training parameters from config
    root_dir = config["root_dir"]
    where_to_save_checkpoint_path = config["where_to_save_checkpoint_path"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    resnet = config["resnet"]

    # Load data
    train_dataloader, test_dataloader = load_data(
        root_dir, image_size, batch_size
    )

    # Initialize model
    classes = get_classes()
    net = initialize_model(resnet, transfer, len(classes))

    # Check for checkpoint and load
    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print(f"Loading checkpoint '{checkpoint}'")
            net.load_state_dict(torch.load(checkpoint))
        else:
            print(
                f"No checkpoint found at '{checkpoint}', starting training from scratch"
            )

    # Train model
    losses, accuracies, epochs_list = train_evaluate_model(
        net,
        train_dataloader,
        test_dataloader,
        epochs,
        learning_rate,
        where_to_save_checkpoint_path,
    )

    # Compute accuracy by class
    all_labels, all_predictions = load_test_pred_labels(
        net, test_dataloader, classes
    )

    # Calculate precision and recall (NEED TO REVISE TO GRACE'S VERSION)
    prec_dict, rec_dict, total_accuracy_val = compute_precision_recall_accur(
        all_labels, all_predictions
    )

    # Print and plot results
    plot_results(epochs_list, losses, accuracies, out_dir)
    print_summary(
        root_dir,
        where_to_save_checkpoint_path,
        image_size,
        batch_size,
        epochs,
        learning_rate,
        resnet,
        out_dir,
        transfer,
        checkpoint,
        accuracies,
        losses,
        epochs_list,
        prec_dict,
        rec_dict,
        total_accuracy_val,
    )


def load_data(root_dir, image_size, batch_size):
    """
    Loads training and testing datasets using the DumitruxDataset class.

    This function applies transformations to the datasets and wraps them in DataLoader
    instances for batch processing.

    Args:
        root_dir (str): Directory where the datasets are located.
        image_size (int): The size to which the images will be resized.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: A tuple containing DataLoader instances for both training and testing datasets.
    """

    ROOT_DIR = root_dir

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    training_dataset = DumitruxDataset(
        ROOT_DIR,
        train=True,
        download=False,
        transform=transform,
        fix_split=True,
    )
    testing_dataset = DumitruxDataset(
        ROOT_DIR,
        train=False,
        download=False,
        transform=transform,
        fix_split=True,
    )

    train_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def initialize_model(resnet, transfer, num_classes):
    """
    Initializes a neural network model based on the specified parameters.

    This function supports initializing a custom CNN model or a ResNet model
    (with transfer learning).

    Args:
        resnet (int): Specifies which ResNet model to use (18 or 34) if transfer learning is enabled.
        transfer (bool): Indicates whether to use transfer learning.
        num_classes (int): The number of classes in the output layer.

    Returns:
        net: The initialized neural network model.
    """
    if transfer:
        if resnet == 18:
            print("Implementing transfer learning with ResNet18.")
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif resnet == 34:
            print("Implementing transfer learning with ResNet34.")
            model = models.resnet34(weights="IMAGENET1K_V1")

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print("Implementing custom CNN.")
        model = Net()

    return model


class Net(nn.Module):
    """
    Custom Convolutional Neural Network.

    IMPORTANT NOTE: Only works with an image resolution of 32x32.

    Args:
        nn.Module: PyTorch's base class for all neural network modules.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_evaluate_model(
    net,
    train_dataloader,
    test_dataloader,
    epochs,
    learning_rate,
    where_to_save_checkpoint_path,
):
    """
    Trains and evaluates the neural network model.

    This function runs the training loop over the specified number of epochs,
    calculates loss and accuracy, and saves the model checkpoint.

    Args:
        net (torch.nn.Module): The neural network model to train.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the testing dataset.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        where_to_save_checkpoint_path (str): Path to save the model checkpoint.

    Returns:
        tuple: A tuple containing lists of losses, accuracies, and epoch numbers.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    losses = []
    accuracies = []
    epochs_list = []

    # Loop through epochs
    for epoch in range(epochs):
        print("epoch = ", epoch)
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, torch.tensor(labels))
            loss.backward()
            optimizer.step()

        # PATH = where_to_save_checkpoint_path
        torch.save(net.state_dict(), where_to_save_checkpoint_path)

        # Compute loss and accuracy for this epoch
        losses, accuracies, epochs_list = compute_accuracy_each_epoch(
            net, epoch, loss, test_dataloader, accuracies, losses, epochs_list
        )

    return losses, accuracies, epochs_list


def compute_accuracy_each_epoch(
    net, epoch, loss, test_dataloader, accuracies, losses, epochs_list
):
    """
    Computes and stores accuracy, loss, and epoch information for each training epoch.

    This function is designed to be called inside the training loop. It evaluates
    the model on the test dataset and updates the lists of metrics.

    Args:
        net (torch.nn.Module): The trained model.
        epoch (int): The current epoch number.
        loss (torch.Tensor): The loss value for the current epoch.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        accuracies (list): List to store accuracy values.
        losses (list): List to store loss values.
        epochs_list (list): List to store epoch numbers.

    Returns:
        tuple: Updated lists of losses, accuracies, and epoch numbers.
    """
    # Calculate accuracy on testing dataset
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data

            # calculate outputs by running images through the network
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if epoch % 5 == 0:
        accuracies.append(100 * correct / total)
        losses.append(loss.item())
        epochs_list.append(epoch)

    print(
        f"Accuracy of the network on the housing dataset test images: {100 * correct // total} %"
    )

    return losses, accuracies, epochs_list


def load_test_pred_labels(net, test_dataloader, classes):
    """
    Computes the accuracy of the model for each class in the dataset.

    This function iterates through the test dataset and calculates the
    accuracy per class based on the model predictions.

    Args:
        net (torch.nn.Module): The trained model.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        classes (tuple): Tuple containing the class names.

    Returns:
        tuple: Numpy arrays of all true labels and predictions.
    """
    # prepare to count predictions for each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # Prepare arrays to store true labels and predictions
    all_labels = []
    all_predictions = []

    # again no gradients needed
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            # collect the correct predictions for each class
            # for label, prediction in zip(labels, predictions):
            #    if label == prediction:
            #        correct_pred[classes[label]] += 1
            #    total_pred[classes[label]] += 1

    # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #    accuracy = 100 * float(correct_count) / total_pred[classname]
    # print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    print("all_labels looks like: ", all_labels)
    print("length of all_labels = ", len(all_labels))
    print("all_predictions looks like: ", all_predictions)
    print("length of all_predictions: ", len(all_predictions))

    return all_labels, all_predictions


def get_classes():
    """
    Defines and returns the list of class names in the Dumitrux Dataset.

    Returns:
        tuple: A tuple containing the class names as strings.
    """
    classes = (
        "Achaemenid_architecture",
        "American craftsman style",
        "American Foursquare architecture",
        "Ancient Egyptian architecture",
        "Art Deco architecture",
        "Art Nouveau architecture",
        "Baroque architecture",
        "Bauhaus architecture",
        "Beaux-Arts architecture",
        "Byzantine architecture",
        "Chicago school architecture",
        "Colonial architecture",
        "Deconstructivism",
        "Edwardian architecture",
        "Georgian architecture",
        "Gothic architecture",
        "Greek Revival architecture",
        "International style",
        "Novelty architecture",
        "Palladian architecture",
        "Postmodern architecture",
        "Queen Anne architecture",
        "Romanesque architecture",
        "Russian Revival architecture",
        "Tudor Revival architecture",
    )

    return classes


def compute_precision_recall_accur(all_labels, all_predictions):
    """
    Computes the precision and recall metrics for the model.

    Args:
        all_labels (list): The list of true labels from the test dataset.
        all_predictions (list): The list of predicted labels by the model.

    Returns:
        None
    """

    prec_dict = {}
    rec_dict = {}
    total_accuracy_val = None

    if np.array_equal(np.unique(all_labels), np.unique(all_predictions)):
        prec_dict = precision_per_class(all_labels, all_predictions)
        rec_dict = recall_per_class(all_labels, all_predictions)
        total_accuracy_val = total_accuracy(all_labels, all_predictions)
        print("Precision: ", prec_dict)
        print("Rec_dict: ", rec_dict)
        print("Total accuracy val: ", total_accuracy_val)
    else:
        print(
            "The predicted classes contain one or more classes that do not exist in the set of actual classes (or vice versa)"
        )

    return prec_dict, rec_dict, total_accuracy_val


def plot_results(epochs_list, losses, accuracies, out_dir):
    """
    Plots the training loss and accuracy over epochs.

    This function creates and saves a plot showing how loss and accuracy change
    with each epoch during training.

    Args:
        epochs_list (list): List of epoch numbers.
        losses (list): List of loss values per epoch.
        accuracies (list): List of accuracy values per epoch.
        out_dir (str): Directory to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, losses, label="Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, accuracies, label="Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "training_performance.png"))


def print_summary(
    root_dir,
    where_to_save_checkpoint_path,
    image_size,
    batch_size,
    epochs,
    learning_rate,
    resnet,
    out_dir,
    transfer,
    checkpoint,
    accuracies,
    losses,
    epochs_list,
    prec_dict,
    rec_dict,
    total_accuracy_val,
):
    """
    Prints a summary of the training parameters and results.

    This function consolidates and displays the training configuration and
    the outcomes of the model training, such as accuracy, loss, precision, and recall.

    Args:
        root_dir (str): Directory of the dataset.
        where_to_save_checkpoint_path (str): Path where the model checkpoint is saved.
        image_size (int): Size of the images used for training.
        batch_size (int): Batch size used in training.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        resnet (int): ResNet model version used.
        out_dir (str): Output directory for plots and results.
        transfer (bool): Indicates if transfer learning was used.
        checkpoint (str): Path of the used checkpoint.
        accuracies (list): List of accuracies per epoch.
        losses (list): List of losses per epoch.
        epochs_list (list): List of epoch numbers.
        precision (dictionary): Precision metric.
        recall (dictionary): Recall metric.
        total_accuracy(float): Total accuracy metric.

    Returns:
        None
    """
    print("\n Parameters for this run: ")
    print("Root directory: ", root_dir)
    print("where_to_save_checkpoint_path", where_to_save_checkpoint_path)
    print("image_size: ", image_size)
    print("batch_size: ", batch_size)
    print("epochs: ", epochs)
    print("learning_rate: ", learning_rate)
    print("resnet: ", resnet)
    print("out_dir: ", out_dir)
    print("transfer: ", transfer)
    print("checkpoint: ", checkpoint)

    print("\n Results from this run:")
    print("accuracies: ", accuracies)
    print("losses: ", losses)
    print("epochs_list: ", epochs_list)
    print("Precision: ", prec_dict)
    print("Rec_dict: ", rec_dict)
    print("Total accuracy val: ", total_accuracy_val)


if __name__ == "__main__":
    # Set default config file path
    cur_dir = Path(__file__).resolve().parent
    cfg_default = cur_dir / "baseline_config.json"

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=cfg_default,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="baseline_results",
        help="Path to the output directory.",
        required=True,
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Flag to use transfer learning or not. If this flag is used, transfer learning is enabled.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved model checkpoint. If provided, training will continue from this checkpoint.",
    )

    args = parser.parse_args()

    # Unpack config file
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    main(config, args.out_dir, args.transfer, args.checkpoint)
