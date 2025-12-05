import csv
import os
import re
from pathlib import Path

from PIL import Image

# Root directory where your images and _annotations.csv are stored
TRAIN_DIR = Path(r"E:\GIT REPOS\ccpd\train")
ANNOT_FILE = TRAIN_DIR / "_annotations.csv"
OUTPUT_DIR = TRAIN_DIR / "plates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex to extract plate text from filenames like:
# car-wbs-MH01DE2780_00000_png.rf.0e78392e00cad0fff5c58cb25725ebde.jpg
PLATE_REGEX = re.compile(r"car-[^-]+-([A-Z0-9]+)_", re.IGNORECASE)


def load_annotations(csv_path: Path):
    """
    Load _annotations.csv into a dict:
    { filename: (xmin, ymin, xmax, ymax) }
    """
    if not csv_path.exists():
        print(f"Annotation file not found: {csv_path}")
        return {}

    ann = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            try:
                xmin = int(float(row["xmin"]))
                ymin = int(float(row["ymin"]))
                xmax = int(float(row["xmax"]))
                ymax = int(float(row["ymax"]))
            except (KeyError, ValueError):
                continue
            ann[fname] = (xmin, ymin, xmax, ymax)

    print(f"Loaded {len(ann)} annotations from {csv_path.name}")
    return ann


def extract_plate_text(filename: str) -> str | None:
    """
    Extract plate number (e.g. MH01DE2780) from the image filename.
    """
    m = PLATE_REGEX.search(filename)
    if not m:
        return None
    return m.group(1).upper()


def main():
    annotations = load_annotations(ANNOT_FILE)

    output_csv = TRAIN_DIR / "plate_crops.csv"
    fieldnames = [
        "orig_filename",
        "crop_filename",
        "plate_text",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    num_processed = 0
    num_missing_ann = 0
    num_missing_plate = 0

    with output_csv.open("w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        for img_name in os.listdir(TRAIN_DIR):
            # Only consider 'car-' images; you already removed 'video*'
            if not img_name.lower().startswith("car-"):
                continue

            img_path = TRAIN_DIR / img_name

            # Skip non-image files just in case
            if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                continue

            plate_text = extract_plate_text(img_name)
            if not plate_text:
                num_missing_plate += 1
                continue  # no plate text found in filename

            # Look up bounding box from _annotations.csv
            if img_name not in annotations:
                num_missing_ann += 1
                continue  # no annotation for this file

            xmin, ymin, xmax, ymax = annotations[img_name]

            # Crop the image
            try:
                with Image.open(img_path) as im:
                    # Clamp coordinates to image bounds for safety
                    w, h = im.size
                    left = max(0, min(xmin, w))
                    top = max(0, min(ymin, h))
                    right = max(0, min(xmax, w))
                    bottom = max(0, min(ymax, h))

                    if right <= left or bottom <= top:
                        continue

                    crop = im.crop((left, top, right, bottom))

                    # Create a crop filename
                    crop_filename = f"plate_{plate_text}_{num_processed:06d}.jpg"
                    crop_path = OUTPUT_DIR / crop_filename
                    crop.save(crop_path)

                writer.writerow(
                    {
                        "orig_filename": img_name,
                        "crop_filename": crop_filename,
                        "plate_text": plate_text,
                        "xmin": left,
                        "ymin": top,
                        "xmax": right,
                        "ymax": bottom,
                    }
                )

                num_processed += 1

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"Done. Cropped plates: {num_processed}")
    print(f"Images skipped (no plate text in name): {num_missing_plate}")
    print(f"Images skipped (no annotation): {num_missing_ann}")
    print(f"CSV written to: {output_csv}")
    print(f"Crops saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()