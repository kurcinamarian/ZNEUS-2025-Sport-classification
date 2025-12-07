from pathlib import Path
from model import CNN1,  CNN2,  CNN3

# Model
MODEL = CNN2()                  #CNN1(), CNN2(), CNN3()

# Training parameters
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 10

# Output directory
OUT_DIR = Path("runs") / "sports_cnn_v1"

# Weights & Biases
USE_WANDB = True
WANDB_PROJECT = "zneus-project-2"
WANDB_ENTITY = "ZNEUS-Diabetes"
WANDB_RUN_NAME = "sports_cnn_model3_1"
WANDB_LOG_FREQUENCY = 500