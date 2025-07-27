# Number Plate Recognition using YOLOv8 and CRNN

This project implements a complete pipeline for automatic number plate recognition (ANPR) from images. It uses a YOLOv8 model for detecting number plates and a custom-trained CRNN (Convolutional Recurrent Neural Network) model for recognizing the characters on the detected plates. The entire pipeline is made accessible through a user-friendly Gradio web interface.

## Features

- **YOLOv8 for Detection**: Utilizes a fine-tuned YOLOv8 model for robust and accurate number plate detection.
- **CRNN for OCR**: Employs a CRNN architecture for precise optical character recognition on the cropped number plates.
- **End-to-End Pipeline**: A seamless script (`number_plate_recognition_pipeline.py`) that integrates detection and recognition.
- **Interactive Web App**: A Gradio interface (`gradio_app.py`) that allows users to upload images and see the results in real-time.
- **Training Notebooks**: Jupyter notebooks are provided for training both the YOLOv8 detector and the CRNN OCR model from scratch.

## Project Structure

```
.
├── models/
│   ├── yolov8_3e.pt            # Trained YOLOv8 model
│   └── best_ocr_model.pth      # Trained CRNN OCR model
├── images/
│   └── ...                     # Sample images for testing
├── results/
│   └── ...                     # Output images from the pipeline
├── notebook/
│   ├── train_yolov8_model.ipynb # Notebook for training the detector
│   └── train_ocr_model.ipynb    # Notebook for training the OCR model
├── ocr_model_test.py           # Script to test the OCR model
├── test_yolo_model.py          # Script to test the YOLO model
├── number_plate_recognition_pipeline.py # Main script for the end-to-end pipeline
├── gradio_app.py               # Gradio web interface
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- OpenCV

### Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:zenitsu0509/YOLO-CRNN-Plate-Recognition.git
    cd YOLO-CRNN-Plate-Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    ```

## How to Use

### 1. Run the Gradio Web App (Easiest)

The most straightforward way to use the project is by launching the Gradio interface.

```bash
python gradio_app.py
```

This will start a local web server. Open the provided URL in your browser, upload an image, and see the magic happen!

### 2. Run the Command-Line Pipeline

To process a single image and save the result to the `results/` directory, use the main pipeline script.

```bash
python number_plate_recognition_pipeline.py
```

You can change the image being processed by editing the `IMAGE_TO_PROCESS` variable inside the script.

## Training Your Own Models

The notebooks in the `notebook/` directory provide a step-by-step guide to training your own models.

### 1. Training the YOLOv8 Detector

-   Open and run the `notebook/train_yolov8_model.ipynb` notebook.
-   Make sure your dataset is in the format expected by YOLOv8.
-   The trained model will be saved, and you can replace the existing `yolov8_1e.pt` with your new model.

### 2. Training the CRNN OCR Model

-   Open and run the `notebook/train_ocr_model.ipynb` notebook.
-   Prepare your dataset of cropped number plate images with corresponding labels.
-   The notebook will handle data loading, training, and saving the best model.
-   Replace `best_ocr_model.pth` with your newly trained model.

## Dataset

The models included in this repository were trained on the **CCPD (Chinese City Parking Dataset)**, a large and comprehensive dataset containing over 100,000 images.

The raw and processed datasets used for training are available on Hugging Face:

-   **Full 100k CCPD Dataset**: [zenitsu09/ccpd-100k-yolo](https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo)
    -   This is the main dataset containing the original 100,000 images.

-   **YOLO Detection Dataset**: [zenitsu09/ccpd-yolo-detection](https://huggingface.co/datasets/zenitsu09/ccpd-yolo-detection)
    -   This dataset was derived from the full set and is formatted specifically for training the YOLOv8 detection model.

-   **OCR Recognition Dataset**: [zenitsu09/ccpd-ocr-recognition](https://huggingface.co/datasets/zenitsu09/ccpd-ocr-recognition)
    -   This dataset contains cropped number plate images and their corresponding text labels, created from the full set for training the CRNN OCR model.

## Acknowledgements

- This project uses the powerful [YOLOv8](https://github.com/ultralytics/ultralytics) object detection model.
- The user interface is built with [Gradio](https://www.gradio.app/).
