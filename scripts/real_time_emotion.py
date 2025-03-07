import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models

# Emotion labels for classification (4 classes)
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        
        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Replace the final fully connected layer to output 4 classes
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.6),  # Dropout for regularization
            nn.Linear(512, len(EMOTIONS))  # Output layer for 4 classes
        )

    def forward(self, x):
        return self.resnet(x)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    
    # Convert to PIL Image
    pil_image = Image.fromarray(resized)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization to match training
    ])
    
    # Add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    
    return tensor

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model = EmotionNet().to(device)
    model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
    model.eval()
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face for emotion detection
            face_tensor = preprocess_image(face_roi).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                emotion_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0][emotion_idx].item()
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display emotion and confidence
            emotion_text = f"{EMOTIONS[emotion_idx]}: {confidence:.2f}"
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Emotion Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()