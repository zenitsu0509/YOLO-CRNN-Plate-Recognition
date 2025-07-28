import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from ultralytics import YOLO
import os
import sys
import json
from pathlib import Path
import time
from collections import defaultdict
import re

# CCPD character mappings (excluding province characters since your model doesn't predict them)
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# --- OCR Model (CRNN) ---
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

    def predict(self, image_np):
        image_tensor = self.transform(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        predicted_text = decode_prediction(output, self.idx_map)
        return predicted_text[0]

class NumberPlateDetector:
    def __init__(self, model_path, confidence_threshold=0.1):  # Lower default threshold
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"YOLO Detector loaded successfully from {model_path}")
        print(f"Using confidence threshold: {confidence_threshold}")

    def detect_plates(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, []
        
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    detections.append({'bbox': (x1, y1, x2, y2), 'confidence': confidence})
        
        return image, detections

def parse_ccpd_filename(filename):
    """
    Parse CCPD filename to extract ground truth license plate information.
    Example format: test_0000_0156429597702-90_93-193,486_410,567-419,555_197,567_202,482_424,470-0_0_12_30_24_32_29-116-20_ccpd_base_071283.jpg
    The license plate characters are the 5th part (index 4), separated by hyphens.
    """
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        
        if len(parts) < 5:
            # print(f"Warning: Could not parse {filename}, not enough parts.")
            return None
        
        # The 5th part (index 4) contains the character indices
        nums_part = parts[4]
        char_indices = nums_part.split('_')
        
        if len(char_indices) != 7:
            # print(f"Warning: Could not parse {filename}, incorrect number of character indices.")
            return None
        
        indices = [int(idx) for idx in char_indices]
        
        gt_chars = []
        # Second character (alphabet)
        if 0 <= indices[1] < len(alphabets):
            gt_chars.append(alphabets[indices[1]])
        
        # Last 5 characters (alphanumeric)
        for i in range(2, 7):
            if 0 <= indices[i] < len(ads):
                gt_chars.append(ads[indices[i]])
        
        return ''.join(gt_chars)
        
    except (ValueError, IndexError) as e:
        # print(f"Warning: Error parsing filename '{filename}': {e}")
        return None

def calculate_character_accuracy(predicted, ground_truth):
    """Calculate character-level accuracy"""
    if not predicted or not ground_truth:
        return 0.0
    
    # Align strings by length (pad shorter one)
    max_len = max(len(predicted), len(ground_truth))
    pred_padded = predicted.ljust(max_len, ' ')
    gt_padded = ground_truth.ljust(max_len, ' ')
    
    correct = sum(1 for p, g in zip(pred_padded, gt_padded) if p == g)
    return correct / max_len

def calculate_sequence_accuracy(predicted, ground_truth):
    """Calculate sequence-level accuracy (exact match)"""
    return 1.0 if predicted == ground_truth else 0.0

def clean_predicted_text(text):
    """Clean predicted text by removing invalid characters and spaces"""
    # Remove spaces and convert to uppercase
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned

class CCPDTester:
    def __init__(self, yolo_model_path, ocr_model_path, confidence_threshold=0.1):
        print("Initializing CCPD Pipeline Tester...")
        self.detector = NumberPlateDetector(yolo_model_path, confidence_threshold=confidence_threshold)
        self.ocr = OCRModel(ocr_model_path)
        self.results = {
            'total_images': 0,
            'detection_results': {
                'detected': 0,
                'not_detected': 0,
                'detection_rate': 0.0
            },
            'ocr_results': {
                'total_predictions': 0,
                'character_accuracy': 0.0,
                'sequence_accuracy': 0.0,
                'character_accuracies': [],
                'sequence_accuracies': []
            },
            'processing_times': {
                'detection_times': [],
                'ocr_times': [],
                'total_times': []
            },
            'detailed_results': []
        }
    
    def test_single_image(self, image_path):
        """Test pipeline on a single image"""
        filename = os.path.basename(image_path)
        
        # Parse ground truth from filename
        ground_truth = parse_ccpd_filename(filename)
        if ground_truth is None:
            print(f"Warning: Could not parse ground truth for {filename}. Skipping.")
            return None
        
        start_time = time.time()
        
        # Detection phase
        detection_start = time.time()
        original_image, detections = self.detector.detect_plates(image_path)
        detection_time = time.time() - detection_start
        
        if original_image is None:
            return None
        
        result = {
            'filename': filename,
            'ground_truth': ground_truth,
            'detections': len(detections),
            'detection_time': detection_time,
            'ocr_time': 0.0,
            'total_time': 0.0,
            'predictions': [],
            'best_prediction': '',
            'character_accuracy': 0.0,
            'sequence_accuracy': 0.0
        }
        
        if len(detections) == 0:
            result['total_time'] = time.time() - start_time
            return result
        
        # OCR phase - test on best detection (highest confidence)
        best_detection = max(detections, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_detection['bbox']
        
        # Crop the number plate
        padding = 5
        plate_crop = original_image[max(0, y1-padding):min(original_image.shape[0], y2+padding), 
                                   max(0, x1-padding):min(original_image.shape[1], x2+padding)]
        
        if plate_crop.size > 0:
            plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            
            ocr_start = time.time()
            predicted_text = self.ocr.predict(plate_crop_rgb)
            ocr_time = time.time() - ocr_start
            
            # Clean predicted text
            cleaned_prediction = clean_predicted_text(predicted_text)
            
            result['ocr_time'] = ocr_time
            result['predictions'] = [{'text': predicted_text, 'cleaned': cleaned_prediction, 'confidence': best_detection['confidence']}]
            result['best_prediction'] = cleaned_prediction
            
            # Calculate accuracies
            result['character_accuracy'] = calculate_character_accuracy(cleaned_prediction, ground_truth)
            result['sequence_accuracy'] = calculate_sequence_accuracy(cleaned_prediction, ground_truth)
        
        result['total_time'] = time.time() - start_time
        return result
    
    def test_dataset(self, test_images_dir, max_images=None, save_results=True):
        """Test pipeline on entire dataset"""
        print(f"Starting evaluation on CCPD test dataset...")
        print(f"Test images directory: {test_images_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(test_images_dir).glob(f'*{ext}'))
            image_files.extend(Path(test_images_dir).glob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} test images")
        
        self.results['total_images'] = len(image_files)
        
        # Process each image
        for i, image_path in enumerate(image_files):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
            
            result = self.test_single_image(str(image_path))
            if result is None:
                continue
            
            # Update statistics
            self.results['detailed_results'].append(result)
            
            # Detection statistics
            if result['detections'] > 0:
                self.results['detection_results']['detected'] += 1
            else:
                self.results['detection_results']['not_detected'] += 1
            
            # OCR statistics
            if result['best_prediction']:
                self.results['ocr_results']['total_predictions'] += 1
                self.results['ocr_results']['character_accuracies'].append(result['character_accuracy'])
                self.results['ocr_results']['sequence_accuracies'].append(result['sequence_accuracy'])
            
            # Timing statistics
            self.results['processing_times']['detection_times'].append(result['detection_time'])
            self.results['processing_times']['ocr_times'].append(result['ocr_time'])
            self.results['processing_times']['total_times'].append(result['total_time'])
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Print results
        self._print_results()
        
        # Save results
        if save_results:
            self._save_results(test_images_dir)
        
        return self.results
    
    def _calculate_final_metrics(self):
        """Calculate final evaluation metrics"""
        total_images = len(self.results['detailed_results'])
        
        if total_images > 0:
            # Detection rate
            detected = self.results['detection_results']['detected']
            self.results['detection_results']['detection_rate'] = detected / total_images
            
            # OCR accuracies
            char_accs = self.results['ocr_results']['character_accuracies']
            seq_accs = self.results['ocr_results']['sequence_accuracies']
            
            if char_accs:
                self.results['ocr_results']['character_accuracy'] = np.mean(char_accs)
            if seq_accs:
                self.results['ocr_results']['sequence_accuracy'] = np.mean(seq_accs)
    
    def _print_results(self):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("CCPD PIPELINE EVALUATION RESULTS")
        print("="*60)
        
        # Detection Results
        print(f"\n📊 DETECTION PERFORMANCE:")
        print(f"   Total Images: {self.results['total_images']}")
        print(f"   Detected: {self.results['detection_results']['detected']}")
        print(f"   Not Detected: {self.results['detection_results']['not_detected']}")
        print(f"   Detection Rate: {self.results['detection_results']['detection_rate']:.3f}")
        
        # OCR Results
        print(f"\n🔤 OCR PERFORMANCE:")
        print(f"   Total Predictions: {self.results['ocr_results']['total_predictions']}")
        print(f"   Character Accuracy: {self.results['ocr_results']['character_accuracy']:.3f}")
        print(f"   Sequence Accuracy: {self.results['ocr_results']['sequence_accuracy']:.3f}")
        
        # Timing Results
        times = self.results['processing_times']
        if times['total_times']:
            print(f"\n⏱️  TIMING PERFORMANCE:")
            print(f"   Avg Detection Time: {np.mean(times['detection_times']):.3f}s")
            print(f"   Avg OCR Time: {np.mean(times['ocr_times']):.3f}s")
            print(f"   Avg Total Time: {np.mean(times['total_times']):.3f}s")
        
        # Sample Results
        print(f"\n📝 SAMPLE RESULTS:")
        detailed = self.results['detailed_results']
        for i, result in enumerate(detailed[:5]):
            print(f"   {i+1}. GT: '{result['ground_truth']}' | Pred: '{result['best_prediction']}' | Char Acc: {result['character_accuracy']:.3f}")
        
        print("\n" + "="*60)
    
    def _save_results(self, test_dir):
        """Save detailed results to JSON file"""
        output_file = os.path.join(os.path.dirname(test_dir), 'ccpd_evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_copy = self.results.copy()
        for key in ['character_accuracies', 'sequence_accuracies', 'detection_times', 'ocr_times', 'total_times']:
            if key in results_copy['ocr_results']:
                results_copy['ocr_results'][key] = [float(x) for x in results_copy['ocr_results'][key]]
            if key in results_copy['processing_times']:
                results_copy['processing_times'][key] = [float(x) for x in results_copy['processing_times'][key]]
        
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"📄 Detailed results saved to: {output_file}")

def test_single_known_image():
    """Test with a known working image first"""
    YOLO_MODEL_PATH = 'models/yolov8_3e.pt'
    
    # Test different confidence thresholds
    for threshold in [0.01, 0.1, 0.3, 0.5]:
        print(f"\n=== Testing with confidence threshold: {threshold} ===")
        detector = NumberPlateDetector(YOLO_MODEL_PATH, confidence_threshold=threshold)
        
        # Test with first image from test directory
        test_image = '1k_test_img/test_0000_0156429597702-90_93-193,486_410,567-419,555_197,567_202,482_424,470-0_0_12_30_24_32_29-116-20_ccpd_base_071283.jpg'
        
        if os.path.exists(test_image):
            print(f"Testing image: {test_image}")
            image, detections = detector.detect_plates(test_image)
            print(f"Detections found: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"  Detection {i+1}: bbox={det['bbox']}, confidence={det['confidence']:.3f}")
        else:
            print(f"Test image not found: {test_image}")

def main():
    # Configuration
    YOLO_MODEL_PATH = 'models/yolov8_3e.pt'
    OCR_MODEL_PATH = 'models/best_ocr_model.pth'
    TEST_IMAGES_DIR = '1k_test_img'  # Directory with test images
    MAX_IMAGES = None  # Set to a number to limit testing, None for all images
    
    print("CCPD Pipeline Performance Tester")
    print("="*50)
    print(f"YOLO Model: {YOLO_MODEL_PATH}")
    print(f"OCR Model: {OCR_MODEL_PATH}")
    print(f"Test Directory: {TEST_IMAGES_DIR}")
    print()
    
    # Check if paths exist
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        return
    
    if not os.path.exists(OCR_MODEL_PATH):
        print(f"Error: OCR model not found at {OCR_MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Error: Test images directory not found at {TEST_IMAGES_DIR}")
        return
    
    # Initialize tester and run evaluation
    tester = CCPDTester(YOLO_MODEL_PATH, OCR_MODEL_PATH, confidence_threshold=0.1)
    results = tester.test_dataset(TEST_IMAGES_DIR, max_images=MAX_IMAGES)
    
    print(f"\n✅ Evaluation completed!")

if __name__ == '__main__':
    # First test with a single image to debug detection issues
    # test_single_known_image()
    
    # Run full evaluation 
    main()