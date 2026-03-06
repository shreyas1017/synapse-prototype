"""
OCR module using EasyOCR with improved preprocessing pipeline.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Tuple
import re


class OCRModule:
    """Wrapper for EasyOCR with enhanced preprocessing and filtering."""

    def __init__(self, languages: List[str] = ['en'], gpu: bool = False,
                 min_confidence: float = 0.3):
        """
        Initialize OCR reader.

        Args:
            languages: List of language codes
            gpu: Use GPU if available
            min_confidence: Minimum confidence threshold (0-1)
        """
        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence

        print(f"[OCR] Initializing EasyOCR for languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print(f"[OCR] Ready! GPU={gpu}, Min confidence={min_confidence}")

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing pipeline for better OCR accuracy.

        Steps:
        1. Upscale (small text becomes readable)
        2. Grayscale
        3. CLAHE contrast enhancement
        4. Adaptive thresholding (handles uneven lighting)
        5. Denoise
        6. Sharpen
        """
        # Step 1: Upscale by 1.5x (helps with small text)
        h, w = frame.shape[:2]
        upscaled = cv2.resize(
            frame, (int(w * 1.5), int(h * 1.5)),
            interpolation=cv2.INTER_CUBIC
        )

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # Step 3: CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 4: Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Step 5: Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def _clean_text(self, text: str) -> str:
        """
        Clean and filter extracted text.

        - Remove single characters (usually noise)
        - Remove strings with too many special characters
        - Strip extra whitespace
        """
        if not text:
            return ""

        # Remove isolated single characters (noise)
        words = text.split()
        filtered = [w for w in words if len(w) > 1]

        # Remove tokens that are mostly non-alphanumeric
        cleaned = []
        for word in filtered:
            alnum_count = sum(c.isalnum() for c in word)
            if alnum_count / max(len(word), 1) > 0.5:
                cleaned.append(word)

        return ' '.join(cleaned).strip()

    def extract_text(self, frame: np.ndarray, verbose: bool = True,
                     use_preprocessing: bool = True) -> Tuple[str, List[dict]]:
        """
        Extract text from image with improved accuracy.

        Args:
            frame: Input image (BGR from OpenCV)
            verbose: Print detection info
            use_preprocessing: Apply enhancement pipeline

        Returns:
            Tuple of (cleaned_text, detections_list)
        """
        # Preprocess
        if use_preprocessing:
            processed = self.preprocess_image(frame)
        else:
            processed = frame

        # Run OCR with both detail levels for better coverage
        results = self.reader.readtext(
            processed,
            detail=1,
            paragraph=False,      # Get individual text regions
            width_ths=0.7,        # Merge nearby text boxes
            contrast_ths=0.1,
            adjust_contrast=0.5
        )

        # Parse and filter results
        detections = []
        text_parts = []

        for (bbox, text, confidence) in results:
            if confidence < self.min_confidence:
                continue

            # Clean individual text
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            detections.append({
                'bbox': bbox,
                'text': cleaned,
                'confidence': confidence,
                'raw_text': text
            })
            text_parts.append(cleaned)

        # Combine and final clean
        combined_text = ' '.join(text_parts)
        combined_text = self._clean_text(combined_text)

        if verbose:
            print(f"[OCR] Found {len(detections)} text regions")
            if combined_text:
                print(f"[OCR] Extracted: {combined_text}")
            else:
                print("[OCR] No text detected")

        return combined_text, detections

    def draw_text_boxes(self, frame: np.ndarray,
                        detections: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected text with confidence scores.
        """
        annotated = frame.copy()

        for det in detections:
            bbox = det['bbox']
            text = det['text']
            conf = det['confidence']

            points = np.array(bbox, dtype=np.int32)

            # Color by confidence: green=high, yellow=medium, red=low
            if conf >= 0.7:
                color = (0, 255, 0)
            elif conf >= 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            # Draw polygon box
            cv2.polylines(annotated, [points], True, color, 2)

            # Draw label
            label = f"{text} ({conf:.2f})"
            cv2.putText(
                annotated,
                label,
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return annotated
