"""
Note that this is not yet connected to the rest of the image processor, for now it is a standalone script and will work to test images - Krish
Fixes: white balance, gamma correction, brightness/contrast normalization, denoising.
In Terminal try: python preprocessing.py <image path>
python3 image-processor/preprocessing.py image-processor/images/smores.jpg

I still have to tweak it to actually process it positively, but for now at least its working; just not good
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Auto white balance using Gray World algorithm."""
    result = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result, axis=(0, 1))
    gray_avg = (avg_b + avg_g + avg_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (gray_avg / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (gray_avg / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (gray_avg / (avg_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction. gamma > 1 brightens, < 1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def auto_gamma(img: np.ndarray) -> np.ndarray:
    """Estimate gamma from image brightness and correct."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    target = 127.0
    gamma = np.log(target / 255.0) / np.log(mean_brightness / 255.0 + 1e-6)
    gamma = np.clip(gamma, 0.5, 2.5)
    return gamma_correction(img, gamma)


def normalize_brightness_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def preprocess(img: np.ndarray) -> np.ndarray:
    img = denoise(img)
    img = gray_world_white_balance(img)
    img = auto_gamma(img)
    img = normalize_brightness_contrast(img)
    return img


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_image>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: Could not load image: {path}")
        sys.exit(1)

    processed = preprocess(img)

    out_path = path.parent / f"{path.stem}_processed{path.suffix}"
    cv2.imwrite(str(out_path), processed)
    print(f"Saved processed image to: {out_path}")


if __name__ == "__main__":
    main()
