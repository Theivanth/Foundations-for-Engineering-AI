# Citations and third-party materials

## Dataset

Immanuel0405. *YOLO Object Detection Dataset (office supplies).* Kaggle.
https://www.kaggle.com/datasets/immanuel0405/python-object-detection-dataset-office-supplies

Dataset license: [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA
4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). Cite the dataset
creator and review the license terms before using or sharing the dataset or
trained weights.

The dataset contains YOLO annotation IDs but does not provide a class-name
mapping file. Display names in this project were inferred from labeled sample
images and mapped as follows:

| Source class ID | Display name |
| ---: | --- |
| 15 | Tape dispenser |
| 16 | Highlighter |
| 17 | Glue stick |
| 18 | Pen |
| 19 | Scissors |
| 20 | Protractor |
| 21 | Correction tape |

## Model and training framework

Ultralytics. *YOLO11.* https://docs.ultralytics.com/models/yolo11/

Ultralytics. *Licensing.* https://www.ultralytics.com/license

The training run starts from the pretrained `yolo11n.pt` checkpoint and
fine-tunes it on the dataset above. The Ultralytics software/checkpoint terms
and the dataset terms are independent of this repository's MIT license.

## Training run

Theivanth Rajamohan. *MLE1003 Office Supplies Detection - 50 Images.* Kaggle
Code, version 7. https://www.kaggle.com/code/theivanth/mle1003-office-supplies-detection-50-images

## Software

- Ultralytics Python package: https://pypi.org/project/ultralytics/
- OpenCV Python package: https://pypi.org/project/opencv-python/
- Kaggle API client: https://github.com/Kaggle/kaggle-api
