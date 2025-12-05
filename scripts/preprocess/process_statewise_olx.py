import csv
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


ROOT_DIR = Path(r"E:\GIT REPOS\ccpd\State-wise_OLX")
OUTPUT_DIR = ROOT_DIR / "plates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_tag = root.find("filename")
    if filename_tag is None or not filename_tag.text:
        return None

    obj = root.find("object")
    if obj is None:
        return None

    name_tag = obj.find("name")
    bnd = obj.find("bndbox")
    if name_tag is None or bnd is None or not name_tag.text:
        return None

    plate_text = name_tag.text.strip().upper()

    def _get_int(tag_name):
        t = bnd.find(tag_name)
        if t is None or t.text is None:
            raise ValueError(f"Missing {tag_name} in {xml_path}")
        return int(float(t.text))

    xmin = _get_int("xmin")
    ymin = _get_int("ymin")
    xmax = _get_int("xmax")
    ymax = _get_int("ymax")

    return filename_tag.text.strip(), plate_text, xmin, ymin, xmax, ymax


def main():
    output_csv = ROOT_DIR / "statewise_olx_plate_crops.csv"
    fieldnames = [
        "state",
        "orig_filename",
        "crop_filename",
        "plate_text",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    num_processed = 0

    with output_csv.open("w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        # Each subfolder (AN, AP, MH, ...) contains jpg+xml pairs
        for state_dir_name in os.listdir(ROOT_DIR):
            state_dir = ROOT_DIR / state_dir_name
            if not state_dir.is_dir():
                continue

            for xml_name in os.listdir(state_dir):
                if not xml_name.lower().endswith(".xml"):
                    continue

                xml_path = state_dir / xml_name

                try:
                    parsed = parse_xml(xml_path)
                except Exception as e:
                    print(f"Error parsing {xml_path}: {e}")
                    continue

                if parsed is None:
                    continue

                img_filename, plate_text, xmin, ymin, xmax, ymax = parsed

                img_path = state_dir / img_filename
                if not img_path.exists():
                    print(f"Image not found for XML {xml_path}: {img_filename}")
                    continue

                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                        left = max(0, min(xmin, w))
                        top = max(0, min(ymin, h))
                        right = max(0, min(xmax, w))
                        bottom = max(0, min(ymax, h))

                        if right <= left or bottom <= top:
                            print(f"Invalid bbox in {xml_path}")
                            continue

                        crop = im.crop((left, top, right, bottom))

                        crop_filename = (
                            f"olx_{state_dir_name}_{plate_text}_{num_processed:06d}.jpg"
                        )
                        crop_path = OUTPUT_DIR / crop_filename
                        crop.save(crop_path)

                    writer.writerow(
                        {
                            "state": state_dir_name,
                            "orig_filename": img_filename,
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
                    print(f"Error processing {img_path}: {e}")

    print(f"Done. Cropped plates from State-wise_OLX: {num_processed}")
    print(f"CSV written to: {output_csv}")
    print(f"Crops saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
