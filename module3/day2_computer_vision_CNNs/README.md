# Day 2: Computer Vision with CNNs

This project trains a convolutional neural network to classify casting
product images as defective or OK. It includes training, evaluation, a
Tkinter image-classification app, and an optional KaggleHub dataset download.

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)
- Tkinter (included with the standard Windows Python installer)

Install a PyTorch build compatible with your machine and its CPU/GPU setup.
The standard PyTorch installation page provides the appropriate command for
your platform.

## Set up

Use the existing `m1` environment; this project does not need a separate
virtual environment. Open PowerShell in this folder and install dependencies
into `m1`:

```powershell
$PYTHON = "$HOME\.venvs\m1\Scripts\python.exe"
& $PYTHON -m pip install -r requirements.txt
```

Use `& $PYTHON script.py` for the commands below as well.

The training scripts look for a dataset with this layout:

```text
images/
+-- .../
    +-- train/
    |   +-- def_front/
    |   +-- ok_front/
    +-- test/
        +-- def_front/
        +-- ok_front/
```

The dataset is discovered recursively. If it is not already present, download
it from Kaggle and copy it into this folder:

```powershell
& $PYTHON generate.py
```

This step needs internet access and KaggleHub access to the public
`ravirajsinh45/real-life-industrial-dataset-of-casting-product` dataset.

## Train and evaluate

Train the CNN and save its best validation checkpoint:

```powershell
& $PYTHON train.py
```

Evaluate the saved checkpoint against the test set:

```powershell
& $PYTHON evaluate.py
```

The saved checkpoint is `best_defect_cnn.pth`. Training and evaluation require
the labeled image dataset above.

## Use the image-classification app

Launch the graphical app:

```powershell
& $PYTHON predict.py
```

1. Choose an image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`,
   `.tiff`, or `.webp`).
2. Review the preview and image dimensions.
3. Select **Classify image** and wait for the result and confidence estimate.
4. Use **Clear** to choose another image.

The app uses the saved checkpoint and can classify a selected image without
the training dataset. If `best_defect_cnn.pth` is missing, run training first.
Predictions are model estimates and should be reviewed before taking action.
