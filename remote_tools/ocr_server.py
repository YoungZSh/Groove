#!/usr/bin/env python3
"""Single-concurrency PaddleOCR HTTP worker for the remote Analyzer."""

from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import paddle
from paddleocr import PaddleOCR
from PIL import Image


ROOT = os.environ.get("OCR_MODEL_ROOT", "/data4/yzs/model_cache/paddleocr/whl")
ocr = PaddleOCR(
    lang="en",
    use_gpu=True,
    use_angle_cls=False,
    show_log=False,
    det_limit_side_len=4096,
    det_db_box_thresh=0.2,
    det_model_dir=f"{ROOT}/det/en/en_PP-OCRv3_det_infer",
    rec_model_dir=f"{ROOT}/rec/en/en_PP-OCRv4_rec_infer",
    cls_model_dir=f"{ROOT}/cls/ch_ppocr_mobile_v2.0_cls_infer",
)


def normalize_bbox(value, size):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must be [x1,y1,x2,y2]")
    width, height = size
    x1, y1, x2, y2 = [float(item) for item in value]
    box = [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("invalid bbox")
    area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    if area < 0.1:
        bw, bh = box[2] - box[0], box[3] - box[1]
        box = [max(0, box[0] - 2 * bw), max(0, box[1] - 2 * bh), min(width, box[2] + 2 * bw), min(height, box[3] + 2 * bh)]
    return box


def read_text(payload):
    image = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))).convert("RGB")
    box = normalize_bbox(payload.get("bbox"), image.size)
    origin_x, origin_y = 0, 0
    working = image
    if box:
        origin_x, origin_y = box[0], box[1]
        working = image.crop(box)
    scale = max(1, min(8, int(payload.get("scale", 4))))
    if scale > 1:
        working = working.resize((working.width * scale, working.height * scale), Image.Resampling.LANCZOS)
    result = ocr.ocr(np.asarray(working), cls=False)
    lines = []
    for page in result:
        for polygon, (text, confidence) in page or []:
            lines.append({
                "text": str(text),
                "confidence": float(confidence),
                "bbox": [[round(float(x) / scale + origin_x, 2), round(float(y) / scale + origin_y, 2)] for x, y in polygon],
            })
    value = {"image_size": list(image.size), "crop_bbox": box, "scale": scale, "text": lines}
    paddle.device.cuda.empty_cache()
    return value


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(200, {"status": "ok", "device": "cuda", "tool": "read_text"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 25 * 1024 * 1024:
                raise ValueError("invalid request size")
            self._reply(200, read_text(json.loads(self.rfile.read(size))))
        except Exception as exc:
            self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format, *args):
        return


HTTPServer(("127.0.0.1", int(os.environ.get("OCR_PORT", "8012"))), Handler).serve_forever()
