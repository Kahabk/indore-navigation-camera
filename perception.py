import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

import config


class FloorSegmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[perception] device: {self.device}")
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(config.MODEL_NAME)
            .to(self.device)
            .eval()
        )
        id2label = self.model.config.id2label
        first = {int(i): name.split(",")[0].strip() for i, name in id2label.items()}
        self.floor_ids = [i for i, n in first.items() if n in config.FLOOR_LABELS]
        self.danger_ids = [i for i, n in first.items() if n in config.DANGER_LABELS]
        print(f"[perception] floor ids: {self.floor_ids}  danger ids: {self.danger_ids}")

        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

        self.avg_floor = None
        self.avg_danger = None

    def _smooth(self, old, new):
        if old is None:
            return new
        a = config.TEMPORAL_ALPHA
        return a * new + (1 - a) * old

    @torch.no_grad()
    def segment(self, frame_bgr):
        """Returns (floor_mask, danger_mask, danger_close)."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (config.MODEL_INPUT_W, config.MODEL_INPUT_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        logits = self.model(pixel_values=x).logits
        labels = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        floor = np.isin(labels, self.floor_ids).astype(np.float32)
        danger = np.isin(labels, self.danger_ids).astype(np.float32)
        self.avg_floor = self._smooth(self.avg_floor, floor)
        self.avg_danger = self._smooth(self.avg_danger, danger)

        k = np.ones((5, 5), np.uint8)
        floor_mask = (self.avg_floor > 0.5).astype(np.uint8) * 255
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, k)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, k)
        danger_mask = (self.avg_danger > 0.5).astype(np.uint8) * 255

        roi = danger_mask[int(0.35 * h):, w // 4: 3 * w // 4]
        danger_close = bool((roi > 0).mean() > config.DANGER_CLOSE_RATIO)
        return floor_mask, danger_mask, danger_close
