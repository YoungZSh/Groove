from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from groove.focus_image import build_focus_image
from groove.schemas import EvidenceImageConfig, InstanceBox, ObjectCrop


class FocusImageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "original.png"
        y, x = np.indices((80, 120))
        self.pixels = np.stack(((x % 2) * 255, (y % 2) * 255, (x * 13 + y * 17) % 256), axis=-1).astype("uint8")
        Image.fromarray(self.pixels).save(self.path)

    def selected(self, box, **kwargs):
        # The renderer must read the original: this crop path does not exist.
        return ObjectCrop(query="private target", score=1.0, raw_box=box,
                          expanded_box=box, path=self.root / "unused.jpg", area_fraction=0.1, **kwargs)

    def render(self, boxes, **kwargs):
        return build_focus_image(self.path, [self.selected(box) for box in boxes],
                                 self.root / "result", EvidenceImageConfig(mode="focus", **kwargs))

    def test_union_preserves_pixels_and_blends_only_exterior_with_red_outlines(self):
        before = self.path.read_bytes()
        boxes = [(10, 10, 45, 45), (30, 30, 65, 65), (85, 15, 110, 55)]
        record = self.render(boxes)
        with Image.open(record.path) as loaded:
            self.assertEqual(loaded.size, (120, 80))
            self.assertEqual(loaded.format, "PNG")
            actual = np.asarray(loaded)
        original = Image.fromarray(self.pixels)
        blurred = np.asarray(original.filter(ImageFilter.GaussianBlur(12)))
        expected_background = np.floor((self.pixels.astype(float) + blurred.astype(float)) / 2).astype("uint8")
        inside = np.zeros((80, 120), dtype=bool)
        outlines = np.zeros_like(inside)
        for x1, y1, x2, y2 in boxes:
            inside[y1:y2, x1:x2] = True
            outlines[y1, x1:x2] = outlines[y2 - 1, x1:x2] = True
            outlines[y1:y2, x1] = outlines[y1:y2, x2 - 1] = True
        np.testing.assert_array_equal(actual[inside & ~outlines], self.pixels[inside & ~outlines])
        np.testing.assert_array_equal(actual[~inside], expected_background[~inside])
        self.assertTrue(np.all(actual[outlines] == (255, 0, 0)))
        # Gap between separated boxes stays blurred; do not fill the union's hull.
        np.testing.assert_array_equal(actual[25, 75], expected_background[25, 75])
        self.assertTrue(np.any(actual[~inside] != self.pixels[~inside]))
        self.assertEqual(record.boxes, boxes)
        self.assertEqual(self.path.read_bytes(), before)

    def test_alpha_endpoints_and_radius_are_effective(self):
        for alpha in (0.0, 0.25, 1.0):
            with self.subTest(alpha=alpha):
                record = self.render([(20, 20, 60, 60)], blur_alpha=alpha, blur_radius=2.0)
                with Image.open(record.path) as image:
                    actual = np.asarray(image)
                blurred = np.asarray(Image.fromarray(self.pixels).filter(ImageFilter.GaussianBlur(2)))
                expected = np.floor((1 - alpha) * self.pixels.astype(float) + alpha * blurred).astype("uint8")
                np.testing.assert_array_equal(actual[:10], expected[:10])

    def test_border_boxes_clip_without_off_by_one_and_full_frame_stays_clear(self):
        record = self.render([(-20, -10, 150, 90)])
        self.assertEqual(record.boxes, [(0, 0, 120, 80)])
        with Image.open(record.path) as image:
            actual = np.asarray(image)
        np.testing.assert_array_equal(actual[1:-1, 1:-1], self.pixels[1:-1, 1:-1])
        self.assertTrue(np.all(actual[0] == (255, 0, 0)))
        self.assertTrue(np.all(actual[:, -1] == (255, 0, 0)))

    def test_instance_evidence_uses_individual_boxes_without_deduplication(self):
        boxes = [(10.2, 10.2, 35.1, 35.1), (20.5, 20.5, 45.0, 45.0)]
        selected = self.selected((0, 0, 120, 80), kind="instance_boxes",
                                 instances=[InstanceBox(bbox=box, score=0.9) for box in boxes])
        record = build_focus_image(self.path, [selected], self.root / "instances",
                                   EvidenceImageConfig(mode="focus"))
        self.assertEqual(record.boxes, [(10, 10, 36, 36), (20, 20, 45, 45)])
        with Image.open(record.path) as image:
            self.assertNotEqual(image.getpixel((70, 70)), tuple(self.pixels[70, 70]))

    def test_no_visible_boxes_fails_and_original_cannot_be_overwritten(self):
        for boxes in ([], [(200, 100, 210, 110)], [(10, 10, 5, 20)]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                self.render(boxes)
        source = self.root / "focus.png"
        source.write_bytes(self.path.read_bytes())
        before = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            build_focus_image(source, [self.selected((0, 0, 10, 10))], self.root,
                              EvidenceImageConfig(mode="focus"))
        self.assertEqual(source.read_bytes(), before)

    def test_invalid_settings_rejected_before_rendering(self):
        for settings in ({"mode": "unknown"}, {"blur_alpha": -0.1}, {"blur_alpha": 1.1},
                         {"blur_alpha": float("nan")}, {"blur_radius": 0},
                         {"blur_radius": -1}, {"blur_radius": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                EvidenceImageConfig(**settings)


if __name__ == "__main__":
    unittest.main()
