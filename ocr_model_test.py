import torch
import cv2
from torchvision import transforms
import torch.nn as nn
import sys
import numpy as np

# --- 1. Define the CRNN Model Architecture ---
# This must be the *exact* same architecture as used for training.
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
        # Dummy forward pass to get the feature size
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
        # The output for inference should be [Batch, Seq, Classes]
        # The training output is permuted for CTC loss, so we don't do that here.
        return output

# --- 2. Define the Decoder ---
def decode_prediction(preds, idx_map):
    """
    Decodes the raw output of the CRNN model using a greedy approach.
    """
    # The model output is [Batch, Seq, Classes], which is what we need
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    decoded_texts = []
    for pred in preds:
        sequence = []
        # CTC decoding: collapse blanks and duplicates
        for i in range(len(pred)):
            char_index = pred[i]
            if char_index != 0 and (i == 0 or char_index != pred[i-1]):
                sequence.append(idx_map.get(char_index, '?'))
        decoded_texts.append("".join(sequence))
    return decoded_texts


# --- 3. Main Prediction Function ---
def predict_image(image_path, model_path):
    """
    Loads the model and performs prediction on a single image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    char_map = checkpoint['char_map']
    idx_map = {v: k for k, v in char_map.items()}
    vocab_size = len(char_map) + 1

    model = CRNN(vocab_size).to(device)
    # The state_dict keys should now match perfectly
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"cv2.imread returned None for {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
    
    predicted_text = decode_prediction(output, idx_map)
    
    return predicted_text[0]


if __name__ == '__main__':
    # --- Configuration ---
    # Make sure this path is correct
    MODEL_PATH = 'models/best_ocr_model_ctc.pth' 
    

    IMAGE_PATH = "images/plate1.jpg"
    predicted_plate = predict_image(IMAGE_PATH, MODEL_PATH)

    print("-" * 30)
    print(f"Image Path: {IMAGE_PATH}")
    print(f"Predicted Number Plate: {predicted_plate}")
    print("-" * 30)
