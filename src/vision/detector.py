"""
YOLO26-Nano detector wrapper for real-time object detection.
Uses NCNN for inference (ARM NEON optimized, Pi 4 compatible).
"""

import ncnn
import cv2
import numpy as np
from typing import List, Dict

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


class YOLODetector:
    """NCNN-based wrapper for YOLO26 object detection (Pi 4 optimized)."""

    def __init__(self, model_path: str = "yolo26n_ncnn_320", device: str = "cpu",
                 confidence: float = 0.35, iou: float = 0.4,
                 detection_size: int = None, num_threads: int = 4):
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.num_threads = num_threads

        if detection_size is not None:
            self.detection_size = detection_size
        elif "256" in model_path:
            self.detection_size = 256
        elif "320" in model_path:
            self.detection_size = 320
        elif "480" in model_path:
            self.detection_size = 480
        else:
            self.detection_size = 640

        print(f"[DETECTOR] Loading YOLO26-Nano from {model_path}...")
        print(f"[DETECTOR] Detection size: {self.detection_size}x{self.detection_size}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.load_param(f"{model_path}/model.ncnn.param")
        self.net.load_model(f"{model_path}/model.ncnn.bin")

        import os, yaml
        meta_path = f"{model_path}/metadata.yaml"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            names_raw = meta.get("names", {})
            self.names = names_raw if isinstance(names_raw, dict) else {i: n for i, n in enumerate(names_raw)}
        else:
            self.names = {i: n for i, n in enumerate(COCO_NAMES)}

        print(f"[DETECTOR] Model loaded on cpu via NCNN ({num_threads} threads)")
        print(f"[DETECTOR] Confidence threshold: {confidence}, IoU: {iou}")

    def _nms(self, boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        return keep

    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[Dict]:
        orig_h, orig_w = frame.shape[:2]
        size = self.detection_size

        scale = size / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        mat_in = ncnn.Mat.from_pixels(padded, ncnn.Mat.PixelType.PIXEL_BGR, size, size)
        mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")

        out = np.array(mat_out).T
        detections = []
        if out.size == 0:
            return detections

        boxes_cxcywh = out[:, :4]
        class_scores = out[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.confidence
        if not np.any(mask):
            return detections

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        cx, cy, bw, bh = boxes_cxcywh[:,0], boxes_cxcywh[:,1], boxes_cxcywh[:,2], boxes_cxcywh[:,3]
        boxes_xyxy = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)

        keep = self._nms(boxes_xyxy, confidences, self.iou)
        for i in keep:
            rx1 = int(max(0, min(boxes_xyxy[i,0] / scale, orig_w)))
            ry1 = int(max(0, min(boxes_xyxy[i,1] / scale, orig_h)))
            rx2 = int(max(0, min(boxes_xyxy[i,2] / scale, orig_w)))
            ry2 = int(max(0, min(boxes_xyxy[i,3] / scale, orig_h)))
            cls_id = int(class_ids[i])
            detections.append({
                "bbox": [rx1, ry1, rx2, ry2],
                "confidence": float(confidences[i]),
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, str(cls_id))
            })

        if verbose and detections:
            print(f"[DETECTOR] Found {len(detections)} objects")

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = det["class_name"]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return annotated_frame

    def get_model_info(self) -> Dict:
        return {
            "model": self.model_path,
            "device": self.device,
            "detection_size": self.detection_size,
            "names": self.names,
            "num_classes": len(self.names)
        }
