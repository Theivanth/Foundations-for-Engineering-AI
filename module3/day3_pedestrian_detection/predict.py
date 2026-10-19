import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from transformer import get_device


DEFAULT_SCORE_THRESHOLD = 0.65
PERSON_CLASS_ID = 1
IMAGE_TRANSFORM = transforms.ToTensor()
INFERENCE_MIN_SIZE = 480
INFERENCE_MAX_SIZE = 960
NMS_THRESHOLD = 0.70


def load_model() -> tuple[torch.nn.Module, torch.device]:
    device = get_device()
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device)
    model.transform.min_size = (INFERENCE_MIN_SIZE,)
    model.transform.max_size = INFERENCE_MAX_SIZE
    model.roi_heads.nms_thresh = NMS_THRESHOLD
    model.eval()
    return model, device


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("The confidence threshold must be between 0 and 1.")

    image_tensor = IMAGE_TRANSFORM(image.convert("RGB")).to(device)
    with torch.inference_mode():
        output = model([image_tensor])[0]

    keep = (output["labels"] == PERSON_CLASS_ID) & (
        output["scores"] >= score_threshold
    )
    return (
        output["boxes"][keep].detach().cpu(),
        output["scores"][keep].detach().cpu(),
    )


class PedestrianDetectorApp:
    def __init__(
        self,
        root: tk.Tk,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.root = root
        self.model = model
        self.device = device
        self.image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.score_threshold = tk.DoubleVar(value=DEFAULT_SCORE_THRESHOLD)

        self.root.title("Pedestrian Detection System")
        self.root.geometry("700x800")
        self.root.configure(bg="#f3f4f6")

        tk.Label(
            root,
            text="PEDESTRIAN DETECTOR",
            font=("Arial", 20, "bold"),
            bg="#f3f4f6",
            fg="#111827",
        ).pack(pady=18)
        tk.Label(
            root,
            text=f"COCO-pretrained Faster R-CNN MobileNet | {device}",
            font=("Arial", 10),
            bg="#f3f4f6",
            fg="#6b7280",
        ).pack()
        tk.Label(
            root,
            text="1. Choose a photo   2. Click Detect Pedestrians   3. Read the count",
            font=("Arial", 11),
            bg="#f3f4f6",
            fg="#374151",
        ).pack(pady=(12, 0))

        self.image_label = tk.Label(
            root,
            text="No image selected",
            bg="#e5e7eb",
            width=60,
            height=20,
        )
        self.image_label.pack(pady=16)

        tk.Button(
            root,
            text="1. CHOOSE PHOTO",
            command=self.upload_image,
            font=("Arial", 12, "bold"),
            bg="#2563eb",
            fg="white",
            padx=24,
            pady=10,
        ).pack(pady=8)
        tk.Button(
            root,
            text="2. DETECT PEDESTRIANS",
            command=self.inspect_image,
            font=("Arial", 12, "bold"),
            bg="#374151",
            fg="white",
            padx=24,
            pady=10,
        ).pack(pady=8)

        self.threshold_label = tk.Label(
            root,
            text=f"Confidence threshold: {DEFAULT_SCORE_THRESHOLD:.0%}",
            font=("Arial", 10),
            bg="#f3f4f6",
            fg="#374151",
        )
        self.threshold_label.pack(pady=(8, 0))
        tk.Scale(
            root,
            from_=0.10,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.score_threshold,
            command=self.update_threshold_label,
            length=320,
            bg="#f3f4f6",
            highlightthickness=0,
        ).pack()

        self.result_label = tk.Label(
            root,
            text="Result: Not inspected",
            font=("Arial", 16, "bold"),
            bg="#f3f4f6",
            fg="#111827",
        )
        self.result_label.pack(pady=16)

        tk.Label(
            root,
            text="Research / Educational Use Only",
            font=("Arial", 9),
            bg="#f3f4f6",
            fg="#6b7280",
        ).pack(pady=18)

    def update_threshold_label(self, value: str) -> None:
        self.threshold_label.configure(
            text=f"Confidence threshold: {float(value):.0%}"
        )

    def show_image(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((560, 340))
        self.tk_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.tk_image, text="", width=560, height=360)

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            with Image.open(file_path) as source:
                self.image = source.convert("RGB")
            self.show_image(self.image)
            self.result_label.configure(
                text="Photo ready — click Detect Pedestrians",
                fg="#111827",
            )
        except (OSError, ValueError) as error:
            self.image = None
            messagebox.showerror("Image Error", str(error))

    def inspect_image(self) -> None:
        if self.image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first.",
            )
            return

        try:
            boxes, scores = predict(
                self.image,
                self.model,
                self.device,
                score_threshold=self.score_threshold.get(),
            )
            annotated = self.image.copy()
            draw = ImageDraw.Draw(annotated)
            font = ImageFont.load_default()

            for box, score in zip(boxes.tolist(), scores.tolist()):
                x_min, y_min, x_max, y_max = box
                draw.rectangle(
                    (x_min, y_min, x_max, y_max),
                    outline="#16a34a",
                    width=max(2, annotated.width // 300),
                )
                draw.text(
                    (x_min, max(0, y_min - 16)),
                    f"Pedestrian {score:.0%}",
                    fill="#16a34a",
                    font=font,
                )

            self.show_image(annotated)
            count = len(scores)
            if count:
                self.result_label.configure(
                    text=f"Result: Pedestrians detected — count: {count}",
                    fg="#16a34a",
                )
            else:
                self.result_label.configure(
                    text="Result: No pedestrians detected",
                    fg="#b45309",
                )
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            messagebox.showerror("Prediction Error", str(error))


def main() -> None:
    try:
        model, device = load_model()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        messagebox_root = tk.Tk()
        messagebox_root.withdraw()
        messagebox.showerror(
            "Model Loading Error",
            f"Could not load the COCO-pretrained pedestrian detector.\n\n{error}",
        )
        messagebox_root.destroy()
        raise

    root = tk.Tk()
    PedestrianDetectorApp(root, model, device)
    root.mainloop()


if __name__ == "__main__":
    main()