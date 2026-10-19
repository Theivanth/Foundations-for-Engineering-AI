"""Choose an image in a window and classify it with the trained CNN."""

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from train import DefectCNN
from transformer import IMAGE_SIZE, configure_tk_environment


def readable_class_name(class_name: str) -> str:
    """Convert ImageFolder names such as ``def_front`` to readable labels."""
    normalized = class_name.lower()
    if normalized.startswith("def") or "defect" in normalized:
        return "Defective"
    if normalized.startswith("ok") or "normal" in normalized or "good" in normalized:
        return "OK"
    return class_name.replace("_", " ").title()


def predict_image(image_path: Path, checkpoint_path: Path) -> tuple[str, float]:
    """Return the readable predicted class and its probability (0 to 1)."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Checkpoint is missing a valid class_to_idx mapping.")

    image_size = tuple(checkpoint.get("image_size", IMAGE_SIZE))
    if len(image_size) != 2:
        raise ValueError("Checkpoint image_size must contain height and width.")

    model = DefectCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    # Keep inference preprocessing aligned with training's grayscale/resize/tensor steps.
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        predicted_idx = int(probabilities.argmax().item())

    idx_to_class = {index: name for name, index in class_to_idx.items()}
    if predicted_idx not in idx_to_class:
        raise ValueError(f"Predicted class index {predicted_idx} is absent from checkpoint mapping.")
    return (
        readable_class_name(idx_to_class[predicted_idx]),
        float(probabilities[predicted_idx].item()),
    )


def main() -> None:
    configure_tk_environment()
    root = tk.Tk()
    root.title("Casting Defect Image Classifier")
    root.geometry("760x720")
    root.minsize(580, 620)
    root.configure(background="#f3f5f8")

    checkpoint_path = Path(__file__).resolve().parent / "best_defect_cnn.pth"
    selected_path: Path | None = None
    prediction_results: queue.Queue[tuple[str, str, float]] = queue.Queue()

    style = ttk.Style(root)
    style.configure("App.TFrame", background="#f3f5f8")
    style.configure("Card.TFrame", background="#ffffff")
    style.configure(
        "Title.TLabel", background="#f3f5f8", foreground="#172033",
        font=("Segoe UI", 22, "bold"),
    )
    style.configure(
        "Subtitle.TLabel", background="#f3f5f8", foreground="#586174",
        font=("Segoe UI", 10),
    )
    style.configure(
        "Card.TLabel", background="#ffffff", foreground="#273247",
        font=("Segoe UI", 10),
    )
    style.configure(
        "Preview.TLabel", background="#f7f8fb", foreground="#667085",
        font=("Segoe UI", 11),
    )
    style.configure(
        "Result.TLabel", background="#ffffff", foreground="#586174",
        font=("Segoe UI", 15, "bold"),
    )
    style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 9))
    style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))

    container = ttk.Frame(root, padding=26, style="App.TFrame")
    container.pack(fill="both", expand=True)
    ttk.Label(
        container, text="Casting Defect Classifier", style="Title.TLabel"
    ).pack(anchor="w")
    ttk.Label(
        container,
        text="Select a product image to check whether it is OK or defective.",
        style="Subtitle.TLabel",
    ).pack(anchor="w", pady=(4, 18))

    image_card = ttk.Frame(container, padding=12, style="Card.TFrame")
    image_card.pack(fill="both", expand=True)
    preview = ttk.Label(
        image_card, text="Your image preview will appear here",
        anchor="center", style="Preview.TLabel",
    )
    preview.pack(fill="both", expand=True)

    controls = ttk.Frame(container, style="App.TFrame")
    controls.pack(fill="x", pady=(14, 0))
    file_text = tk.StringVar(value="No image selected")
    ttk.Label(
        controls, textvariable=file_text, style="Subtitle.TLabel",
        wraplength=490, justify="left",
    ).pack(side="left", fill="x", expand=True, padx=(0, 12))
    choose_button = ttk.Button(controls, text="Choose image")
    choose_button.pack(side="right")

    actions = ttk.Frame(container, style="App.TFrame")
    actions.pack(fill="x", pady=(12, 0))
    classify_button = ttk.Button(
        actions, text="Classify image", style="Primary.TButton", state="disabled"
    )
    classify_button.pack(side="left")
    clear_button = ttk.Button(actions, text="Clear")
    clear_button.pack(side="left", padx=(8, 0))
    progress = ttk.Progressbar(actions, mode="indeterminate", length=150)
    progress.pack(side="right", padx=(12, 0))
    status_text = tk.StringVar(value="Choose an image to get started.")
    ttk.Label(
        actions, textvariable=status_text, style="Subtitle.TLabel"
    ).pack(side="right")

    result_card = ttk.Frame(container, padding=16, style="Card.TFrame")
    result_card.pack(fill="x", pady=(16, 0))
    ttk.Label(
        result_card, text="Prediction", style="Card.TLabel"
    ).pack(anchor="w")
    result_text = tk.StringVar(value="Your result will appear here.")
    result_label = ttk.Label(
        result_card, textvariable=result_text, style="Result.TLabel"
    )
    result_label.pack(anchor="w", pady=(8, 4))
    confidence_bar = ttk.Progressbar(result_card, maximum=100, mode="determinate")
    confidence_bar.pack(fill="x", pady=(4, 0))
    confidence_text = tk.StringVar(value="")
    ttk.Label(
        result_card, textvariable=confidence_text, style="Card.TLabel"
    ).pack(anchor="e", pady=(4, 0))

    def choose_image() -> None:
        nonlocal selected_path
        filename = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return

        image_path = Path(filename)
        try:
            with Image.open(image_path) as image:
                display_image = image.convert("RGB")
                image_size = image.size
                display_image.thumbnail((640, 390), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Unable to open image", f"Please choose a valid image file.\n\n{exc}",
                parent=root,
            )
            return

        selected_path = image_path
        preview.configure(image=photo, text="")
        preview.image = photo
        file_text.set(f"{image_path.name}  |  {image_size[0]} x {image_size[1]} pixels")
        classify_button.configure(state="normal")
        status_text.set("Ready to classify.")
        result_text.set("Your result will appear here.")
        result_label.configure(style="Result.TLabel")
        confidence_bar.configure(value=0)
        confidence_text.set("")

    def classify_selected_image() -> None:
        if selected_path is None:
            return

        image_path = selected_path
        choose_button.configure(state="disabled")
        classify_button.configure(state="disabled")
        clear_button.configure(state="disabled")
        progress.start(12)
        status_text.set("Analyzing image…")
        result_text.set("Please wait while the model analyzes your image.")
        confidence_bar.configure(value=0)
        confidence_text.set("")

        def run_prediction() -> None:
            try:
                label, confidence = predict_image(image_path, checkpoint_path)
            except Exception as exc:
                prediction_results.put(("error", str(exc), 0.0))
            else:
                prediction_results.put(("success", label, confidence))

        threading.Thread(target=run_prediction, daemon=True).start()

    def clear_selection() -> None:
        nonlocal selected_path
        selected_path = None
        preview.configure(image="", text="Your image preview will appear here")
        preview.image = None
        file_text.set("No image selected")
        classify_button.configure(state="disabled")
        status_text.set("Choose an image to get started.")
        result_text.set("Your result will appear here.")
        result_label.configure(style="Result.TLabel")
        confidence_bar.configure(value=0)
        confidence_text.set("")

    def show_prediction_result() -> None:
        try:
            result_type, message, confidence = prediction_results.get_nowait()
        except queue.Empty:
            root.after(100, show_prediction_result)
            return

        progress.stop()
        choose_button.configure(state="normal")
        classify_button.configure(state="normal" if selected_path else "disabled")
        clear_button.configure(state="normal")
        if result_type == "error":
            status_text.set("Prediction could not be completed.")
            result_text.set("An error occurred while classifying this image.")
            confidence_bar.configure(value=0)
            messagebox.showerror("Prediction error", message, parent=root)
        else:
            status_text.set("Analysis complete.")
            result_text.set(f"Result: {message}")
            result_label.configure(
                style="PositiveResult.TLabel" if message == "OK"
                else "NegativeResult.TLabel"
            )
            confidence_bar.configure(value=confidence * 100)
            confidence_text.set(f"Confidence: {confidence:.1%}")

        root.after(100, show_prediction_result)

    style.configure(
        "PositiveResult.TLabel", background="#ffffff", foreground="#147d54",
        font=("Segoe UI", 15, "bold"),
    )
    style.configure(
        "NegativeResult.TLabel", background="#ffffff", foreground="#b42318",
        font=("Segoe UI", 15, "bold"),
    )

    choose_button.configure(command=choose_image)
    classify_button.configure(command=classify_selected_image)
    clear_button.configure(command=clear_selection)
    root.after(100, show_prediction_result)

    root.mainloop()


if __name__ == "__main__":
    main()
