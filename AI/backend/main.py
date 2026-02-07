import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenCV Face Detection (Haar Cascade)
# Download the cascade file if it doesn't exist
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mock Model Architecture (Simplified for prototype)
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 112 * 112, 2) # Assuming input size 224x224

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = DeepfakeDetector()
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def root():
    return {"message": "DeepGuard AI Backend is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        # Face Detection using OpenCV
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected in the image.",
                "prediction": "Unknown",
                "confidence": 0
            }

        # For the prototype, we analyze the first detected face
        # In a real app, we'd loop through all faces
        
        # Preprocess for AI model
        input_tensor = transform(image).unsqueeze(0)
        
        # Simulated Inference
        # In a real scenario, we would use: with torch.no_grad(): output = model(input_tensor)
        # For now, we generate a pseudo-random result based on image properties for demonstration
        # (e.g., if image has a lot of high frequency noise, we might flag it)
        
        # Let's simulate a confidence score
        seed = sum(contents[:100]) % 100
        is_fake = seed > 70 # Arbitrary logic for prototype
        confidence = 0.85 + (seed % 15) / 100.0
        
        return {
            "success": True,
            "prediction": "Fake" if is_fake else "Real",
            "confidence": float(confidence),
            "faces_detected": len(faces)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
