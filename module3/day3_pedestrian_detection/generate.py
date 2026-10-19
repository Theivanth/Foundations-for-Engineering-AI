"""Download and prepare Penn-Fudan data for use in a training script.

Import ``get_dataset_paths`` and call it when the training code needs the
dataset. Importing this module does not trigger a download.
"""

from pathlib import Path
import shutil
import zipfile

DATASET_SLUG = "psvishnu/pennfudan-database-for-pedestrian-detection-zip"


def download_dataset() -> Path:
    """Download the Kaggle dataset and return the directory containing its files."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def _find_dataset_root(folder: Path) -> Path | None:
    """Find the folder containing both the image and mask directories."""
    for candidate in (folder, *folder.rglob("*")):
        if candidate.is_dir() and (candidate / "PNGImages").is_dir() and (candidate / "PedMasks").is_dir():
            return candidate
    return None


def extract_dataset(
    download_dir: Path | str,
    destination: Path | str | None = None,
) -> Path:
    """Return the dataset root, extracting ZIP files when necessary."""
    download_dir = Path(download_dir)
    dataset_root = _find_dataset_root(download_dir)
    if dataset_root is not None:
        return dataset_root

    archives = list(download_dir.rglob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"Could not find PNGImages/PedMasks or a ZIP archive under {download_dir}"
        )

    extraction_dir = (
        Path(destination)
        if destination is not None
        else Path(__file__).resolve().parent / "data" / "pennfudan"
    )
    extraction_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(extraction_dir)

    dataset_root = _find_dataset_root(extraction_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Extracted dataset at {extraction_dir}, but PNGImages and PedMasks were not found"
        )
    return dataset_root


def get_dataset_dir() -> Path:
    """Download (if needed) and keep a project-local copy of the dataset."""
    local_dir = Path(__file__).resolve().parent / "data" / "pennfudan"
    existing = _find_dataset_root(local_dir) if local_dir.exists() else None
    if existing is not None:
        return existing

    download_dir = download_dataset()
    downloaded_root = _find_dataset_root(download_dir)
    if downloaded_root is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        # KaggleHub keeps its own cache elsewhere; copy the files into this project.
        if downloaded_root.resolve() != local_dir.resolve():
            shutil.copytree(downloaded_root, local_dir, dirs_exist_ok=True)
        return _find_dataset_root(local_dir) or local_dir

    return extract_dataset(download_dir, destination=local_dir)


def get_dataset_paths() -> tuple[Path, Path, Path]:
    """Return ``(dataset_dir, images_dir, masks_dir)`` for training code."""
    dataset_dir = get_dataset_dir()
    return dataset_dir, dataset_dir / "PNGImages", dataset_dir / "PedMasks"


if __name__ == "__main__":
    dataset_dir, images_dir, masks_dir = get_dataset_paths()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Images directory:  {images_dir}")
    print(f"Masks directory:   {masks_dir}")
