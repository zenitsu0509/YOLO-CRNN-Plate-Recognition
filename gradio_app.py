import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download

# --- 1. OCR Model (Improved CRNN for Indian Number Plates) ---
class CRNN(nn.Module):
    """Improved CRNN Model with deeper architecture for Indian number plate OCR"""
    def __init__(self, vocab_size, hidden_size=256, img_height=32, img_width=160):
        super(CRNN, self).__init__()
        
        # Deeper CNN with more capacity
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
            
            # Block 5 - reduce height to 1
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        
        # Calculate feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_height, img_width)
            cnn_out = self.cnn(dummy_input)
            b, c, h, w = cnn_out.size()
            self.feature_size = c * h
            self.seq_len = w

        # Bidirectional LSTM with more layers
        self.rnn = nn.LSTM(
            self.feature_size, 
            hidden_size, 
            bidirectional=True, 
            num_layers=3,
            batch_first=True, 
            dropout=0.3
        )
        
        # Attention mechanism (must match saved model)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layer with dropout
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w)
        conv = conv.permute(0, 2, 1)  # (B, W, C*H)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(conv)
        
        # Apply dropout and classifier
        rnn_out = self.dropout(rnn_out)
        output = self.classifier(rnn_out)
        
        return output.log_softmax(2).permute(1, 0, 2)  # (T, B, vocab_size)


def decode_predictions(outputs, idx_map):
    """Greedy CTC decoding"""
    _, max_indices = torch.max(outputs, 2)
    decoded = []
    for i in range(max_indices.size(1)):
        raw = max_indices[:, i].cpu().numpy()
        chars = []
        prev = -1
        for idx in raw:
            if idx != 0 and idx != prev:  # not blank and not repeated
                chars.append(idx_map.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    return decoded


class OCRModel:
    """OCR Model wrapper for Indian Number Plate recognition"""
    
    HF_MODEL_REPO = "zenitsu09/indian-plate-ocr-model"
    HF_MODEL_FILE = "best_plate_ocr_model.pth"
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load from local path first, otherwise download from HuggingFace
        if model_path and os.path.exists(model_path):
            checkpoint_path = model_path
            print(f"Loading OCR model from local path: {model_path}")
        else:
            print(f"Downloading OCR model from HuggingFace: {self.HF_MODEL_REPO}")
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self.HF_MODEL_REPO,
                    filename=self.HF_MODEL_FILE
                )
                print(f"Model downloaded to: {checkpoint_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                if model_path:
                    print(f"Falling back to local path: {model_path}")
                    checkpoint_path = model_path
                else:
                    raise FileNotFoundError(f"Could not find or download OCR model")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            print(f"Error: OCR Model file not found at '{checkpoint_path}'")
            sys.exit(1)

        # Load config and character maps
        self.config = checkpoint.get('config', {'img_height': 32, 'img_width': 160})
        self.char_map = checkpoint['char_map']
        self.idx_map = checkpoint['idx_map']
        vocab_size = checkpoint.get('vocab_size', len(self.char_map) + 1)

        # Initialize model with correct dimensions
        self.model = CRNN(
            vocab_size, 
            img_height=self.config.get('img_height', 32),
            img_width=self.config.get('img_width', 160)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Use albumentations for transforms (same as training)
        self.transform = A.Compose([
            A.Resize(self.config.get('img_height', 32), self.config.get('img_width', 160)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"OCR Model loaded successfully on {self.device}")
        print(f"  - Vocab size: {vocab_size}")
        print(f"  - Image size: {self.config.get('img_height', 32)}x{self.config.get('img_width', 160)}")
        if 'char_accuracy' in checkpoint:
            print(f"  - Training char accuracy: {checkpoint['char_accuracy']:.2%}")

    def predict(self, image_np):
        """Predict plate text from numpy image (RGB)"""
        # Ensure image is RGB
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Apply transforms
        augmented = self.transform(image=image_np)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_text = decode_predictions(output, self.idx_map)
        
        return predicted_text[0]

# --- 2. Number Plate Detector (YOLO) ---
class NumberPlateDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"YOLO Detector loaded successfully from {model_path}.")

    def detect_plates(self, image): # Modified to accept an image array
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    detections.append({'bbox': (x1, y1, x2, y2), 'confidence': float(confidence)})
        
        return detections

# --- 3. Gradio Pipeline ---

# Model paths
YOLO_MODEL_PATH = 'models/yolov8_3e.pt'
OCR_MODEL_PATH = 'models/best_plate_ocr_model.pth'  # Will download from HF if not found

# Load models once
print("Loading models...")
detector = NumberPlateDetector(YOLO_MODEL_PATH)
ocr = OCRModel(OCR_MODEL_PATH)  # Auto-downloads from HuggingFace if local file not found
print("Models loaded successfully!")

def recognize_plate(image):
    """
    Main function for the Gradio interface.
    Takes an uploaded image, performs detection and OCR, and returns the annotated image.
    """
    if image is None:
        return None

    try:
        # Ensure image is a proper numpy array
        image = np.array(image)
        
        # Gradio provides images in RGB format, but OpenCV works with BGR.
        original_image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detections = detector.detect_plates(original_image_bgr)
        
        print(f"\nFound {len(detections)} potential number plates.")

        annotated_image = image.copy()  # Work with the RGB image for annotation

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Crop the number plate from the original image (use the RGB version)
            padding = 5
            h, w = image.shape[:2]
            crop_y1 = max(0, y1 - padding)
            crop_y2 = min(h, y2 + padding)
            crop_x1 = max(0, x1 - padding)
            crop_x2 = min(w, x2 + padding)
            
            plate_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if plate_crop.size == 0:
                print(f"  - Skipping detection {i+1} due to empty crop.")
                continue
            
            # OCR model expects an RGB image, which `plate_crop` already is.
            try:
                plate_text = ocr.predict(plate_crop)
            except Exception as e:
                print(f"  - OCR error for detection {i+1}: {e}")
                plate_text = "OCR_ERROR"
            
            print(f"  - Detection {i+1}: BBox={det['bbox']}, Conf={det['confidence']:.2f}, Predicted Text='{plate_text}'")

            # Draw bounding box and predicted text on the image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{plate_text}"
            
            # Position the label above the bounding box
            label_y = y1 - 15 if y1 - 15 > 15 else y1 + 25
            cv2.putText(annotated_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        return annotated_image
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        # Return original image with error text
        error_img = image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return error_img

# --- 4. Launch Gradio Interface ---
if __name__ == "__main__":
    iface = gr.Interface(
        fn=recognize_plate,
        inputs=gr.Image(type="numpy", label="Upload an Image"),
        outputs=gr.Image(type="numpy", label="Result"),
        title="🚗 Indian Number Plate Recognition",
        description="""
        Upload an image to detect and read Indian number plates.
        
        **Models Used:**
        - **Detection:** YOLOv8 for number plate localization
        - **OCR:** Custom CRNN model trained on Indian number plates
        
        The OCR model is automatically downloaded from [HuggingFace](https://huggingface.co/zenitsu09/indian-plate-ocr-model) if not found locally.
        """,
        examples=[
            ['images/image1.jpg'],
            ['images/image2.jpg'],
            ['images/image3.jpg']
        ],
        theme=gr.themes.Soft()
    )
    iface.launch()
