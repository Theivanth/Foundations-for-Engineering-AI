# Office Supplies Video Object Detection

This project runs a fine-tuned YOLO11n object detector on a webcam or video
file. It draws bounding boxes and class names on the video as it runs.

## Detected classes

The displayed names are mapped from the source dataset's numeric class IDs:

| Model class ID | Display name |
| ---: | --- |
| 0 | Tape dispenser |
| 1 | Highlighter |
| 2 | Glue stick |
| 3 | Pen |
| 4 | Scissors |
| 5 | Protractor |
| 6 | Correction tape |

The Kaggle dataset provides YOLO annotation IDs but no class-name YAML. These
names were assigned by inspecting the labeled sample images; they are not an
official class-name list published with the dataset.

## Requirements

- Python 3.10 or newer
- A webcam or a readable video file
- The trained model at `models/best.pt` (downloaded separately; model weights
  are not committed)

Install the dependencies from PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include Kaggle and `python-dotenv` for the training workflow;
they are not needed just to run the video detector.

## Run detection

Start webcam 0:

```powershell
python detect_video.py
```

Use a different webcam, a video file, or a custom model:

```powershell
python detect_video.py --source 1
python detect_video.py --source "C:\path\to\video.mp4"
python detect_video.py --model "C:\path\to\best.pt" --source 0
```

The default confidence threshold is `0.4`. Override it with `--conf`:

```powershell
python detect_video.py --source 0 --conf 0.55
```

Download the latest trained checkpoint from Kaggle before running detection:

```powershell
New-Item -ItemType Directory -Force models
kaggle kernels output theivanth/mle1003-office-supplies-detection-50-images `
  --file-pattern "^best\.pt$" --path models
```

Press **Q** in the video window to stop. Inference currently runs on the CPU.
The application checks that the model has the expected seven classes before
starting.

## Training and validation

Training is configured for Kaggle in [`kaggle_train/`](./kaggle_train/).
`kernel-metadata.json` points to the office-supplies dataset and requests a
Tesla T4 GPU. With the Kaggle CLI installed and authenticated, push a new
training run from this directory:

```powershell
kaggle kernels push --path kaggle_train
```

The script uses up to 50 training images, trains for 20 epochs, and validates
on the validation split. The latest recorded run used 50 training images, 45
validation images, and 30 test images. Its validation results were:

- mAP50: **0.7659**
- mAP50-95: **0.5996**

These are validation metrics, not test-set metrics. The run summary is in
[`outputs/training_summary.json`](./outputs/training_summary.json), and the
training and validation logs are under `outputs/`.

## Data, model, and licensing

The training data comes from the
[YOLO Object Detection Dataset (office supplies)](https://www.kaggle.com/datasets/immanuel0405/python-object-detection-dataset-office-supplies)
by Kaggle user `immanuel0405`, published under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
That license is non-commercial and has attribution and share-alike terms.
Review it before redistributing or using the dataset or fine-tuned weights.

The detector uses Ultralytics YOLO11 and the pretrained `yolo11n.pt` weights.
Ultralytics' software and model terms are separate from this project's MIT
license; see the [Ultralytics license information](https://www.ultralytics.com/license).
The fine-tuned model is not relicensed by the MIT `LICENSE` in this project.

The root [`LICENSE`](./LICENSE) applies only to original project code. See
[`CITATIONS.md`](./CITATIONS.md) for dataset, model, and software references.
