import os

mandatory_folders = [
    'data/volume_files',
    'data/projection_files',
    'data/tif_files',
    'data/nn_models',
    'data/datasets_files'
]

for folder in mandatory_folders:
    if not os.path.isdir('new_folder'):
        os.makedirs(folder, exist_ok=True)