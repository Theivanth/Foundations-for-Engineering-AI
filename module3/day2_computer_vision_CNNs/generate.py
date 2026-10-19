import shutil
from pathlib import Path

import kagglehub

#Program will download a dataset from Kaggle and copy the images to a local directory.
dataset_path = kagglehub.dataset_download(
    "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
)

images_path = Path(__file__).resolve().parent / "images"
shutil.copytree(dataset_path, images_path, dirs_exist_ok=True)

print("Dataset images copied to:", images_path)