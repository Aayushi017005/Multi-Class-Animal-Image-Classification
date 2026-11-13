"""
download_data.py - Dataset Downloader & Sanitizer

Downloads, verifies, and organizes a balanced multi-class animal image dataset
for 5 classes: Butterfly, Cat, Dog, Elephant, Horse.

Features:
- Retrieves high-quality, royalty-free images from Wikimedia Commons.
- Performs perceptual deduplication (dhash) and aspect ratio sanity checks.
- Splits the dataset cleanly and strictly into:
  - dataset/train (70 images per class = 350 total)
  - dataset/val   (15 images per class = 75 total)
  - dataset/test  (15 images per class = 75 total)
"""

import os
import io
import json
import urllib.request
import urllib.parse
from PIL import Image

CLASSES = {
    "butterfly": "butterfly insect lepidoptera",
    "cat": "felis catus domestic cat",
    "dog": "canis familiaris domestic dog",
    "elephant": "elephant wildlife loxodonta",
    "horse": "horse equus caballus",
}

DATASET_DIR = "dataset"
TRAIN_PER_CLASS = 70
VAL_PER_CLASS = 15
TEST_PER_CLASS = 15
TOTAL_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
HEADERS = {"User-Agent": "AnimalClassifierProject/2.0 (educational-use)"}


def dhash(image, hash_size=8):
    """Compute difference hash for perceptual image deduplication."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getflatteneddata() if hasattr(gray, "getflatteneddata") else gray.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
            decimal_value = 0
    return "".join(hex_string)


def hamming_distance(s1, s2):
    return bin(int(s1, 16) ^ int(s2, 16)).count("1")


def is_image_valid(img):
    w, h = img.size
    if w < 100 or h < 100:
        return False
    aspect = w / h
    if aspect < 0.45 or aspect > 2.2:
        return False
    return True


def fetch_image_urls(query, max_results=120):
    search_api = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": "6",
        "srlimit": str(max_results),
        "format": "json"
    }
    url = f"{search_api}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  Error querying search: {e}")
        return []

    results = data.get("query", {}).get("search", [])
    titles = [
        r["title"] for r in results
        if not r["title"].lower().endswith((".svg", ".tif", ".tiff", ".ogg", ".pdf", ".gif"))
    ]

    image_urls = []
    chunk_size = 35
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i:i + chunk_size]
        info_params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "300",
            "format": "json"
        }
        info_url = f"{search_api}?{urllib.parse.urlencode(info_params)}"
        req2 = urllib.request.Request(info_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req2, timeout=15) as resp2:
                info_data = json.loads(resp2.read().decode("utf-8"))
            pages = info_data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                info = pdata.get("imageinfo")
                if info and len(info) > 0:
                    thumb = info[0].get("thumburl") or info[0].get("url")
                    if thumb:
                        image_urls.append(thumb)
        except Exception:
            pass

    return image_urls


def download_image(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if not is_image_valid(img):
            return None
        return img
    except Exception:
        return None


def collect_images_for_class(cls_name, query):
    collected = []
    seen_hashes = []

    # Check existing images
    for split in ["train", "val", "test"]:
        dir_path = os.path.join(DATASET_DIR, split, cls_name)
        if not os.path.exists(dir_path):
            continue
        for f in os.listdir(dir_path):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fp = os.path.join(dir_path, f)
            try:
                with Image.open(fp) as img:
                    img_rgb = img.convert("RGB")
                    if not is_image_valid(img_rgb):
                        continue
                    h = dhash(img_rgb)
                    is_dup = any(hamming_distance(h, sh) <= 4 for sh in seen_hashes)
                    if is_dup:
                        continue
                    seen_hashes.append(h)
                    collected.append(img_rgb.copy())
            except Exception:
                pass

    needed = TOTAL_PER_CLASS - len(collected)
    if needed > 0:
        print(f"  Downloading {needed} more images for '{cls_name}'...")
        urls = fetch_image_urls(query, max_results=120)
        for url in urls:
            if len(collected) >= TOTAL_PER_CLASS:
                break
            img = download_image(url)
            if img is None:
                continue
            h = dhash(img)
            if any(hamming_distance(h, sh) <= 4 for sh in seen_hashes):
                continue
            seen_hashes.append(h)
            collected.append(img)

    return collected[:TOTAL_PER_CLASS]


def setup_dataset():
    print("=" * 60)
    print("Multi-Class Animal Image Classification - Dataset Setup")
    print(f"Splits per class: {TRAIN_PER_CLASS} Train, {VAL_PER_CLASS} Val, {TEST_PER_CLASS} Test")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls_name), exist_ok=True)

    for cls_name, query in CLASSES.items():
        print(f"\nProcessing class: '{cls_name}'")
        imgs = collect_images_for_class(cls_name, query)

        train_imgs = imgs[:TRAIN_PER_CLASS]
        val_imgs = imgs[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
        test_imgs = imgs[TRAIN_PER_CLASS + VAL_PER_CLASS:TOTAL_PER_CLASS]

        for idx, img in enumerate(train_imgs, start=1):
            p = os.path.join(DATASET_DIR, "train", cls_name, f"{cls_name}_{idx:03d}.jpg")
            img.save(p, "JPEG", quality=92)

        for idx, img in enumerate(val_imgs, start=1):
            p = os.path.join(DATASET_DIR, "val", cls_name, f"{cls_name}_{idx:03d}.jpg")
            img.save(p, "JPEG", quality=92)

        for idx, img in enumerate(test_imgs, start=1):
            p = os.path.join(DATASET_DIR, "test", cls_name, f"{cls_name}_{idx:03d}.jpg")
            img.save(p, "JPEG", quality=92)

        print(f"  Saved {cls_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test.")

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    setup_dataset()
