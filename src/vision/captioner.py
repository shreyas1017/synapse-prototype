"""
Scene captioning module using BLIP (Salesforce).
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
import torch


class SceneCaptioner:
    """Wrapper for BLIP image captioning."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cpu", max_length: int = 30, min_length: int = 5):
        """
        Initialize BLIP captioning model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
            max_length: Maximum caption length in tokens
            min_length: Minimum caption length in tokens
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.min_length = min_length
        
        print(f"[CAPTIONER] Loading BLIP model: {model_name}")
        print("[CAPTIONER] This may take 1-2 minutes on first run (downloading ~1GB)...")
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("[CAPTIONER] Running on GPU")
        else:
            self.model = self.model.to("cpu")
            print("[CAPTIONER] Running on CPU")
        
        self.model.eval()  # Set to evaluation mode
        
        print(f"[CAPTIONER] Ready! Max length={max_length}")
    
    def generate_caption(self, frame: np.ndarray, verbose: bool = True) -> str:
        """
        Generate caption for an image.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            verbose: Print caption
            
        Returns:
            Generated caption as string
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Process image
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        
        if verbose:
            print(f"[CAPTIONER] Generated: {caption}")
        
        return caption
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'model': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'min_length': self.min_length
        }
