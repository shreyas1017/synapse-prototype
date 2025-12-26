"""
OCR module using EasyOCR for text extraction.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Tuple, Optional


class OCRModule:
    """Wrapper for EasyOCR text extraction."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False, 
                 min_confidence: float = 0.3):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
            gpu: Use GPU if available
            min_confidence: Minimum confidence threshold (0-1)
        """
        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence
        
        print(f"[OCR] Initializing EasyOCR for languages: {languages}")
        print("[OCR] This may take 1-2 minutes on first run (downloading models)...")
        
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        print(f"[OCR] Ready! GPU={gpu}, Min confidence={min_confidence}")
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            frame: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def extract_text(self, frame: np.ndarray, verbose: bool = True, 
                     use_preprocessing: bool = True) -> Tuple[str, List[dict]]:
        """
        Extract text from image.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            verbose: Print detection info
            use_preprocessing: Apply image enhancement before OCR
            
        Returns:
            Tuple of (combined_text, detections_list)
            detections_list contains: [{'bbox', 'text', 'confidence'}, ...]
        """
        # Preprocess if requested
        if use_preprocessing:
            processed = self.preprocess_image(frame)
        else:
            processed = frame
        
        # Run OCR
        results = self.reader.readtext(processed)
        
        # Parse results
        detections = []
        text_parts = []
        
        for (bbox, text, confidence) in results:
            if confidence >= self.min_confidence:
                detections.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
                text_parts.append(text)
        
        # Combine all text
        combined_text = ' '.join(text_parts)
        
        if verbose:
            print(f"[OCR] Found {len(detections)} text regions")
            if combined_text:
                print(f"[OCR] Extracted: {combined_text}")
        
        return combined_text, detections
    
    def draw_text_boxes(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected text.
        
        Args:
            frame: Input image
            detections: List from extract_text()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            text = det['text']
            conf = det['confidence']
            
            # Convert bbox to integer points
            points = np.array(bbox, dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(annotated, [points], True, (0, 255, 255), 2)
            
            # Draw text label
            label = f"{text} ({conf:.2f})"
            cv2.putText(
                annotated,
                label,
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
        
        return annotated
