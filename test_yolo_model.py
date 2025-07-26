import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from pathlib import Path

class NumberPlateDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the number plate detector
        
        Args:
            model_path (str): Path to your trained YOLO model (.pt file)
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect_plate(self, image_path, save_results=True, output_dir="results"):
        """
        Detect number plates in an image
        
        Args:
            image_path (str): Path to input image
            save_results (bool): Whether to save annotated image
            output_dir (str): Directory to save results
            
        Returns:
            list: List of detection results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
            
        # Run inference
        results = self.model(image_path, conf=self.confidence_threshold)
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Store detection info
                    detection_info = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    }
                    detections.append(detection_info)
                    
                    # Draw bounding box on image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence label
                    label = f'Plate: {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        # Display results
        self.display_results(detections, image_path)
        
        return detections, annotated_image
    
    def display_results(self, detections, image_path):
        """Display detection results"""
        print(f"\n--- Detection Results for {os.path.basename(image_path)} ---")
        print(f"Number of plates detected: {len(detections)}")
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            print(f"Plate {i+1}:")
            print(f"  Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Width: {bbox[2] - bbox[0]}px, Height: {bbox[3] - bbox[1]}px")
    
    def detect_batch(self, image_folder, output_dir="batch_results"):
        """
        Detect plates in multiple images
        
        Args:
            image_folder (str): Path to folder containing images
            output_dir (str): Directory to save results
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f"*{ext}"))
            image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        all_detections = {}
        for image_file in image_files:
            print(f"\nProcessing: {image_file.name}")
            detections, _ = self.detect_plate(str(image_file), save_results=True, output_dir=output_dir)
            all_detections[image_file.name] = detections
        
        # Summary
        total_plates = sum(len(detections) for detections in all_detections.values())
        print(f"\n--- Batch Processing Summary ---")
        print(f"Images processed: {len(image_files)}")
        print(f"Total plates detected: {total_plates}")

# def main():
#     parser = argparse.ArgumentParser(description='YOLO Number Plate Detection')
#     parser.add_argument('--model', required=True, help='Path to trained YOLO model (.pt file)')
#     parser.add_argument('--image', help='Path to input image')
#     parser.add_argument('--folder', help='Path to folder containing images for batch processing')
#     parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
#     parser.add_argument('--output', default='results', help='Output directory for results')
#     parser.add_argument('--no-save', action='store_true', help='Don\'t save annotated images')
    
#     args = parser.parse_args()
    
#     # Validate model file
#     if not os.path.exists(args.model):
#         print(f"Error: Model file not found: {args.model}")
#         return
    
#     # Initialize detector
#     detector = NumberPlateDetector(args.model, args.conf)
    
#     # Process single image or batch
#     if args.image:
#         if os.path.exists(args.image):
#             detector.detect_plate(args.image, save_results=not args.no_save, output_dir=args.output)
#         else:
#             print(f"Error: Image file not found: {args.image}")
#     elif args.folder:
#         if os.path.exists(args.folder):
#             detector.detect_batch(args.folder, args.output)
#         else:
#             print(f"Error: Folder not found: {args.folder}")
#     else:
#         print("Error: Please specify either --image or --folder")

# Example usage functions
# def test_single_image():
#     """Example function to test a single image"""
#     model_path = "models/yolov8_1e.pt"  # Update this path
#     image_path = "images/image1.jpg"    # Update this path
    
#     detector = NumberPlateDetector(model_path, confidence_threshold=0.5)
#     detections, annotated_image = detector.detect_plate(image_path, save_results=True)
    
#     # Optionally display the image
#     cv2.imshow('Number Plate Detection', annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def test_webcam():
    """Example function to test with webcam (real-time detection)"""
    model_path = "models/yolov8_1e.pt"  # Update this path
    
    detector = NumberPlateDetector(model_path, confidence_threshold=0.5)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily and detect
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        detections, annotated_frame = detector.detect_plate(temp_path, save_results=False)
        
        # Display result
        cv2.imshow('Live Number Plate Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

if __name__ == "__main__":
    test_webcam()