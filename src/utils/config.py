from google.colab import drive
import os
import torch

# Mount Google Drive
drive.mount('/content/drive')

# Define the base directory for your project inside Google Drive
BASE_DIR = '/content/drive/My Drive/cifar10_project'

# Ensure the directories exist
OUTPUT_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Data directory: {DATA_DIR}")

BATCH_SIZE = 128
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
LEARNING_RATE = 0.1
BASE_LEARNING_RATE = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')