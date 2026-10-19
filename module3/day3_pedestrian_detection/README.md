# Pedestrian Detection

This project detects and counts people in images using a COCO-pretrained
Faster R-CNN MobileNet model. On compatible systems it uses an Intel GPU
through PyTorch XPU; otherwise it uses CUDA or CPU when available.

## Requirements

- Windows 11
- Python 3.13
- An Intel Arc GPU and current Intel graphics driver for Intel GPU acceleration
- Packages listed in [`requirements.txt`](requirements.txt)

Tkinter is part of the standard Windows Python installation. PyTorch's
pretrained detector weights are downloaded the first time the app starts, so
that first launch requires an internet connection.

## Install

Open PowerShell in this folder and install the dependencies into the Python
environment you want to use:

```powershell
python -m pip install -r requirements.txt
```

For the existing `m1` environment, use:

```powershell
& "$HOME\.venvs\m1\Scripts\python.exe" -m pip install -r requirements.txt
```

The requirements file includes PyTorch's XPU package index. To confirm that
PyTorch can see the Intel GPU:

```powershell
& "$HOME\.venvs\m1\Scripts\python.exe" -c "import torch; print(torch.__version__); print('XPU available:', torch.xpu.is_available())"
```

For GPU use, the output should report an XPU-enabled PyTorch build and
`XPU available: True`. If it reports `False`, update/install the Intel graphics
driver and check the [PyTorch Intel GPU setup guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).

## Launch the graphical app

Double-click **Start Pedestrian Detector.bat** in File Explorer. No terminal
commands are needed for normal use. The included launcher uses the existing
`%USERPROFILE%\.venvs\m1` Python environment. If you installed the packages in
a different environment, edit the `PYTHON` path in the `.bat` file to point to
that environment's `python.exe`.

In the app:

1. Click **Choose Photo** and select a JPG, PNG, or BMP image.
2. Click **Detect Pedestrians**.
3. Read the pedestrian count and inspect the boxes drawn around detections.
4. If needed, adjust the confidence slider and detect again. Lower confidence
   can find more people but may also add false detections.

The app uses the pretrained COCO person detector rather than the project's
custom training checkpoint. It can detect people in general images, but small,
occluded, or unusual views may still be missed, and counts are estimates.

## Train the Penn-Fudan detector

The training and evaluation scripts use the Penn-Fudan Pedestrian dataset in
the `data` folder. The expected layout is:

```text
data/
└── pennfudan/
    ├── PNGImages/
    └── PedMasks/
```

To download and prepare the dataset, configure Kaggle credentials for
KaggleHub, then run:

```powershell
& "$HOME\.venvs\m1\Scripts\python.exe" generate.py
```

Train the custom detector with:

```powershell
& "$HOME\.venvs\m1\Scripts\python.exe" train.py
```

The best checkpoint is saved as `best_pedestrian_detector.pth`. Evaluate it
with:

```powershell
& "$HOME\.venvs\m1\Scripts\python.exe" evaluate.py
```

`predict.py` currently uses the COCO-pretrained detector for general-purpose
image detection. The custom checkpoint produced by `train.py` is not used by
the GUI.
