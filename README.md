# Running the Baseline Model for Architectural Styles Classification

The baseline folder of this repo contains code for running the "baseline" models that provide a point of comparison for the three state-of-the-art models deployed during the Autumn 2023 quarter of the Data Science Clinic. This guide explains how to run the baseline model for classifying architectural styles using ResNet18, ResNet34, or a custom CNN.

## Prerequisites 

- Python 3
- Access to a Unix-like terminal (bash)
- Conda environment (with dependencies installed)

## Dependencies

To re-create the environment used for the baseline model and UNICOM, use the command `conda env create -f env.yml`

## Files Overview

1. `baseline_reproduce.py`: Python script for training and evaluating the model.
2. `train.sh`: Shell script for setting up the environment and running the Python script.
3. `baseline_config.json`: Configuration file with parameters for training.

## Setup Instructions

### Step 1: Create a New Folder

Create a new folder in your desired location: `mkdir <folder name>`. This folder will be used to store output files and error logs.

### Step 2: Update `train.sh`

Edit the `train.sh` file to update the following:

- REQUIRED: Replace both instances of the folder path for output and error logs with the path of the folder you just created.
- REQUIRED: Update the `OUT_DIR` variable with the path of the newly created folder.
- OPTIONAL: Include the `--transfer` tag if you want to implement transfer learning instead of the custom CNN.
- OPTIONAL: If you want to read in a saved model, update the `CHECKPOINT` variable to the path of the model you want to use and include the `--checkpoint ${CHECKPOINT}` tag.

### Step 3: Update `baseline_config.json`

Edit the `baseline_config.json` file:

- Change the `where_to_save_checkpoint_path` to the path where you want to save the model checkpoint. Give the file name a descriptive name so you can distinguish between your different models.
- Update the parameters for the model you want to run. A couple of notes about the limitations on these parameters: (1) The `resnet` variable takes values 18 or 34. (2) The custom CNN only takes an image size and batch size of 32.

### Step 4: Update the Checkpoint

If you are using a pre-trained checkpoint:

- Update the `CHECKPOINT` variable in `train.sh` with the path to your checkpoint file.

If you are not using a pre-trained checkpoint:

- Comment out or remove the `--checkpoint ${CHECKPOINT}` part in `train.sh`.

### Step 5: Either Run the Code via Slurm Cluster or via the terminal

#### Running the code via Slurm Cluster

In the terminal, navigate to the directory containing `train.sh` and run:

```bash
sbatch train.sh
```

#### Step 5: Running the code from the terminal

In the terminal, activate the virtual environment:

```bash
conda activate unicom_env
```

Define the variables `CONFIG-PATH`, `CHECKPOINT`, and `OUT_DIR`:

```bash
CONFIG_PATH="baseline_config.json"
CHECKPOINT="/net/projects/amfam/baseline/<INSERT CHECKPOINT FILE NAME>.pth"
OUT_DIR="<INSERT OUT DIR FILE NAME>"
```

Run: 
```bash
python3 baseline_reproduce.py --config ${CONFIG_PATH} --out_dir ${OUT_DIR} --transfer --checkpoint ${CHECKPOINT}
```

###### Example

```bash
conda activate unicom_env
CONFIG_PATH="baseline_config.json"
CHECKPOINT="/net/projects/amfam/baseline/customCNN_e80_is32.pth"
OUT_DIR="output_customCNN_e80_is32"
python3 baseline_reproduce.py --config ${CONFIG_PATH} --out_dir ${OUT_DIR} --transfer --checkpoint ${CHECKPOINT}
```

## Results

| Model                    | Accuracy | Download Link |
|--------------------------|----------|---------------| 
| Baseline@32px            | 25%      | [Download](https://drive.google.com/uc?export=download&id=1ynNmnAzpzMPL2TGZezTKlGQ6JNReKSTy)    |
| Baseline/ResNet-34@224px | 71%  | [Download](https://drive.google.com/uc?export=download&id=1CAM8M3_l2rwbHEHc0UihiqoxhhmMEpOM)    |
| Baseline/ResNet-18@512px | 73%  | [Download](https://drive.google.com/uc?export=download&id=19NM5R8OjkP79yE_FgKiv-VYtcCIek-Tq)    |

### Additional Experiments

| Model                    | Accuracy |
|--------------------------|----------|
| Baseline@32px            | 25%      |
| Baseline/ResNet-18@32px | 38%  |
| Baseline/ResNet-18@128px | 51%  |
| Baseline/ResNet-18@224px | 60%  |
| Baseline/ResNet-18@256px | 70%  |
| Baseline/ResNet-18@512px | 73%  |
| Baseline/ResNet-34@224px | 71%  |
| Baseline/ResNet-34@256px | 71%  |
| Baseline/ResNet-34@512px | 70%  |


#### Example graph of loss and accuracy
The below graph is from training ResNet-18@224px. This type of graph is produced during each round of training. 

![A line chart illustrating the accuracy and loss throughout each epoch of training](../data/resources/training_performance.png)