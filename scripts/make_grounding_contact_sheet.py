#!/usr/bin/env python3
"""Build a compact overview image from the English Grounding recheck outputs."""

from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "grounding-recheck-english"


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.input_dir.resolve()
    output = (args.output or (out_dir / "contact_sheet.jpg")).resolve()
    paths = sorted(out_dir.glob("*/original_with_english_boxes.jpg"))
    if not paths:
        paths = sorted(out_dir.glob("*/original_with_auto_analyzer_boxes.jpg"))
    if not paths:
        raise SystemExit(f"No annotated images found under {out_dir}")

    columns = 2
    cell_width, image_height, label_height = 720, 430, 42
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for index, path in enumerate(paths):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = row * (image_height + label_height)
        with Image.open(path) as source:
            thumb = fit_image(source, cell_width, image_height)
        sheet.paste(thumb, (x, y))
        label = path.parent.name.replace("step-", "Step ").replace("-", " ")
        draw.rectangle((x, y + image_height, x + cell_width, y + image_height + label_height), fill=(245, 245, 245))
        draw.text((x + 12, y + image_height + 7), label, fill="black", font=font)

    sheet.save(output, quality=92, optimize=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
