import os
from pathlib import Path

GPU_MEM_LIMIT = 16 #GB (real max is 24GB but we let a margin)
GPU_BLOCK_SIZE = (8,8,8)

DATA_FOLDER = Path.cwd() / 'data'

mandatory_folders = [
    'volume_files',
    'projection_files',
    'tif_files',
    'nn_models',
    'datasets_files'
]

for folder in mandatory_folders:
    dir = DATA_FOLDER / folder
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)