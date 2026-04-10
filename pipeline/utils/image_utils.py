import os
import re
from typing import Optional
from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter

# BLOCKED / INVALID IMAGES
BLOCKED_URLS = {
    "https://www.jasminka.cz/images/v/lecenizv.jpg"
}

BLOCKED_PATTERNS = [
    r"lecenizv\.jpg"
]

IMAGE_DIR = "../data/car_images"
AUGMENT_DIR = "../data/car_images_augmented"


def safe_filename(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))


def is_blocked_image(url: str) -> bool:
    if not url:
        return True

    url_clean = url.strip().lower()

    if url_clean in {u.lower() for u in BLOCKED_URLS}:
        return True

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, url_clean, flags=re.IGNORECASE):
            return True

    return False


def get_first_valid_image(image_urls: str) -> Optional[str]:
    """
    From 'url1 | url2 | url3' return first valid non-blocked image.
    """
    if not image_urls or image_urls == "nan":
        return None

    urls = [u.strip() for u in image_urls.split(" | ") if u.strip()]

    for url in urls:
        if not url.startswith("http"):
            continue
        if is_blocked_image(url):
            continue
        return url

    return None


def get_extension_from_url(url: str) -> str:
    url_lower = url.lower()

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if ext in url_lower:
            return ext

    return ".jpg"


def load_image_from_bytes(content: bytes) -> Image.Image:
    return Image.open(BytesIO(content))


def normalize_image(img: Image.Image, max_size=(512, 512)) -> Image.Image:
    """
    Convert image to RGB and resize while preserving aspect ratio.
    """
    img = img.convert("RGB")
    img.thumbnail(max_size)
    return img


def apply_color_tint(img: Image.Image, tint=(20, 0, 0)) -> Image.Image:
    """
    Add a subtle RGB tint.
    Example:
        warm tint -> (20, 10, 0)
        cool tint -> (0, 10, 20)
    """
    tinted = img.copy().convert("RGB")
    pixels = tinted.load()

    for y in range(tinted.height):
        for x in range(tinted.width):
            r, g, b = pixels[x, y]
            r = max(0, min(255, r + tint[0]))
            g = max(0, min(255, g + tint[1]))
            b = max(0, min(255, b + tint[2]))
            pixels[x, y] = (r, g, b)

    return tinted


def generate_variants(img: Image.Image) -> dict[str, Image.Image]:
    """
    Return dictionary of image augmentations.
    """
    return {
        "original": img,
        "bright": ImageEnhance.Brightness(img).enhance(1.25),
        "dark": ImageEnhance.Brightness(img).enhance(0.8),
        "contrast_high": ImageEnhance.Contrast(img).enhance(1.25),
        "contrast_low": ImageEnhance.Contrast(img).enhance(0.85),
        "saturated": ImageEnhance.Color(img).enhance(1.3),
        "desaturated": ImageEnhance.Color(img).enhance(0.7),
        "sharp": ImageEnhance.Sharpness(img).enhance(1.8),
        "blur": img.filter(ImageFilter.GaussianBlur(radius=1)),
        "warm": apply_color_tint(img, (20, 10, 0)),
        "cool": apply_color_tint(img, (0, 10, 20)),
    }


def save_image(img: Image.Image, path: str, quality: int = 92):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="JPEG", quality=quality)


def save_augmented_versions(img: Image.Image, base_name: str, output_dir: str):
    """
    Save all variants into output_dir.
    """
    variants = generate_variants(img)

    for variant_name, variant_img in variants.items():
        out_path = os.path.join(output_dir, f"{base_name}_{variant_name}.jpg")
        save_image(variant_img, out_path)


def find_original_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def find_augmented_image_paths(listing_id):
    """
    Find augmented images
    """
    variants = []
    prefixes = [
        "original", "bright", "dark", "contrast_high", "contrast_low",
        "saturated", "desaturated", "sharp", "blur", "warm", "cool"
    ]

    for variant in prefixes:
        path = os.path.join(AUGMENT_DIR, f"{listing_id}_{variant}.jpg")
        if os.path.exists(path):
            variants.append(path)

    return variants
