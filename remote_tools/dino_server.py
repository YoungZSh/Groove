#!/usr/bin/env python3
"""Single-concurrency Grounding DINO HTTP worker for the remote Analyzer."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_PATH = os.environ.get("DINO_MODEL_PATH", "/data4/yzs/model_cache/grounding-dino-base")
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_PATH, local_files_only=True, dtype=torch.float32
).cuda().eval()
NON_ENGLISH_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def expand_box(box, image_size, margin):
    width, height = image_size
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return [
        max(0, int(x1 - bw * margin)),
        max(0, int(y1 - bh * margin)),
        min(width, int(x2 + bw * margin + 0.999)),
        min(height, int(y2 + bh * margin + 0.999)),
    ]


def ground(payload):
    image = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"]))).convert("RGB")
    query = str(payload["query"]).strip().rstrip(" .") + "."
    if NON_ENGLISH_CJK.search(query):
        raise ValueError("ground_image requires an English query")
    margin = float(payload.get("context_margin", 0.25))
    threshold = float(payload.get("box_threshold", 0.15))
    text_threshold = float(payload.get("text_threshold", 0.15))
    inputs = {k: v.cuda() for k, v in processor(images=image, text=query, return_tensors="pt").items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        target_sizes=[image.size[::-1]],
        threshold=threshold,
        text_threshold=text_threshold,
    )[0]
    if result["scores"].numel() == 0:
        value = {"found": False, "query": query.rstrip("."), "image_size": list(image.size)}
        torch.cuda.empty_cache()
        return value
    index = int(result["scores"].argmax())
    raw = [float(value) for value in result["boxes"][index].tolist()]
    value = {
        "found": True,
        "query": query.rstrip("."),
        "score": float(result["scores"][index]),
        "raw_bbox": [round(value, 2) for value in raw],
        "bbox": expand_box(raw, image.size, margin),
        "image_size": list(image.size),
    }
    del outputs, inputs, result
    torch.cuda.empty_cache()
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
        self._reply(200, {"status": "ok", "device": "cuda", "tool": "ground_image"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 25 * 1024 * 1024:
                raise ValueError("invalid request size")
            self._reply(200, ground(json.loads(self.rfile.read(size))))
        except Exception as exc:
            self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format, *args):
        return


HTTPServer(("127.0.0.1", int(os.environ.get("DINO_PORT", "8011"))), Handler).serve_forever()
