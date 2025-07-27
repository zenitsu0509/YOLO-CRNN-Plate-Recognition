import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from ultralytics import YOLO
import os
import sys

# --- 1. OCR Model (CRNN) ---
# This architecture must be identical to the one used for training the OCR model.
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 128)
            cnn_out = self.cnn(dummy_input)
            b, c, h, w = cnn_out.size()
            feature_size = c * h
        
        self.rnn = nn.LSTM(feature_size, hidden_size, bidirectional=True, num_layers=2, batch_first=True, dropout=0.5)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.contiguous().view(b, c * h, w)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.classifier(rnn_out)
        return output

def decode_prediction(preds, idx_map):
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    decoded_texts = []
    for pred in preds:
        sequence = []
        for i in range(len(pred)):
            char_index = pred[i]
            if char_index != 0 and (i == 0 or char_index != pred[i-1]):
                sequence.append(idx_map.get(char_index, '?'))
        decoded_texts.append("".join(sequence))
    return decoded_texts

class OCRModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            print(f"Error: OCR Model file not found at '{model_path}'")
            sys.exit(1)

        char_map = checkpoint['char_map']
        self.idx_map = {v: k for k, v in char_map.items()}
        vocab_size = len(char_map) + 1

        self.model = CRNN(vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        print(f"OCR Model loaded successfully from {model_path} on {self.device}.")

    def predict(self, image_np):
        image_tensor = self.transform(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        predicted_text = decode_prediction(output, self.idx_map)
        return predicted_text[0]

# --- 2. Number Plate Detector (YOLO) ---
class NumberPlateDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"YOLO Detector loaded successfully from {model_path}.")

    def detect_plates(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, []
        
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    detections.append({'bbox': (x1, y1, x2, y2), 'confidence': float(confidence)})
        
        return image, detections

# --- 3. Main Pipeline ---
def main(yolo_model_path, ocr_model_path, image_path, output_dir="results"):
    # Initialize models
    detector = NumberPlateDetector(yolo_model_path)
    ocr = OCRModel(ocr_model_path)

    # Detect number plates
    original_image, detections = detector.detect_plates(image_path)
    if original_image is None:
        return

    print(f"\nFound {len(detections)} potential number plates in {os.path.basename(image_path)}.")

    annotated_image = original_image.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        
        # Crop the number plate from the original image
        # Add a small buffer to ensure the whole plate is captured
        padding = 5
        plate_crop = original_image[max(0, y1-padding):min(original_image.shape[0], y2+padding), 
                                    max(0, x1-padding):min(original_image.shape[1], x2+padding)]
        
        if plate_crop.size == 0:
            print(f"  - Skipping detection {i+1} due to empty crop.")
            continue

        # Convert cropped plate to RGB for OCR model
        plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        
        # Get OCR prediction
        plate_text = ocr.predict(plate_crop_rgb)
        
        print(f"  - Detection {i+1}: BBox={det['bbox']}, Conf={det['confidence']:.2f}, Predicted Text='{plate_text}'")

        # Draw bounding box and predicted text on the image
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{plate_text}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        # Position the label above the bounding box
        label_y = y1 - 15 if y1 - 15 > 15 else y1 + 25
        cv2.putText(annotated_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Save the final annotated image
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"pipeline_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, annotated_image)
    print(f"\nSaved final annotated image to: {output_path}")


    cv2.imshow('Number Plate Recognition', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- Configuration ---
    YOLO_MODEL_PATH = 'models/yolov8_3e.pt'
    OCR_MODEL_PATH = 'models/best_ocr_model.pth'
    IMAGE_TO_PROCESS = 'images/image.png'

    main(YOLO_MODEL_PATH, OCR_MODEL_PATH, IMAGE_TO_PROCESS)
