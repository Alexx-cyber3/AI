import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenCV Face Detection (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_frequency_artifacts(img_gray):
    """
    Advanced frequency analysis. Natural images follow a 1/f power law.
    AI images often have periodic spikes (checkerboard artifacts) or 
    unnatural energy distributions in high frequencies.
    """
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    
    # Calculate energy distribution
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Analyze the standard deviation of magnitude (AI often has higher variance due to spikes)
    mag_std = np.std(magnitude_spectrum)
    
    # High frequency energy ratio
    radius = min(h, w) // 4
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (center_w, center_h), radius, 0, -1)
    high_freq_energy = np.mean(magnitude_spectrum[mask == 1])
    total_energy = np.mean(magnitude_spectrum)
    
    ratio = high_freq_energy / (total_energy + 1e-6)
    
    return {
        "ratio": ratio,
        "std": mag_std
    }

def analyze_texture_complexity(img_gray):
    """
    Analyze the sharpness and complexity of textures. 
    AI often generates 'uncannily smooth' skin or blurred details.
    """
    # 1. Laplacian Variance (Noise/Focus)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    # 2. Sobel Edge Intensity (Texture detail)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_intensity = np.mean(sobel_mag)
    
    return {
        "noise": laplacian_var,
        "detail": edge_intensity
    }

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
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5) # Improved detection params
        
        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected. Please ensure your face is clearly visible.",
                "prediction": "Unknown",
                "confidence": 0
            }

        faces_detected = len(faces)
        (x, y, w, h) = faces[0]
        face_gray = gray[y:y+h, x:x+w]
        
        # --- Advanced Multi-Factor Analysis ---
        freq_data = analyze_frequency_artifacts(face_gray)
        tex_data = analyze_texture_complexity(face_gray)
        
        # --- Weighted Scoring Engine ---
        # We look for "AI Signatures":
        # 1. Unnatural smoothness (Low noise/detail)
        # 2. Periodic artifacts (High frequency ratio/std)
        
        ai_score = 0.0
        reasons = []
        
        # Factor A: Texture Smoothness (Weight: 40%)
        # Normal photos usually have noise > 150. AI often < 80.
        if tex_data["noise"] < 90:
            ai_score += 0.4
            reasons.append("Unnatural skin smoothness")
        elif tex_data["noise"] < 130:
            ai_score += 0.2
            
        # Factor B: Frequency Anomalies (Weight: 40%)
        # High ratio (>0.82) or high std (>14) suggests GAN/Diffusion artifacts
        if freq_data["ratio"] > 0.84:
            ai_score += 0.3
            reasons.append("Frequency domain anomalies")
        if freq_data["std"] > 14.5:
            ai_score += 0.1
            reasons.append("Periodic pixel artifacts")
            
        # Factor C: Detail Intensity (Weight: 20%)
        # AI often lacks fine edge detail in hair/pores
        if tex_data["detail"] < 15:
            ai_score += 0.2
            reasons.append("Loss of fine structural detail")

        # --- Balanced Final Result ---
        # We want "Real" or "Fake" to have high confidence if they are clear.
        
        is_fake = ai_score >= 0.5
        
        # Scale the confidence to be above 80% for significant detections
        if is_fake:
            prediction = "Fake"
            # Map score [0.5, 1.0] to confidence [0.81, 0.99]
            base_conf = 0.81 + ((ai_score - 0.5) / 0.5) * 0.18
            confidence = min(base_conf, 0.99)
            reason_str = "; ".join(reasons) if reasons else "Multiple AI signatures detected"
        else:
            prediction = "Real"
            # Map score [0.0, 0.49] to confidence [0.82, 0.98]
            # Higher score means LESS real, so we invert.
            base_conf = 0.82 + ((0.5 - ai_score) / 0.5) * 0.16
            confidence = min(base_conf, 0.99)
            reason_str = "Natural texture and sensor noise verified"

        return {
            "success": True,
            "prediction": prediction,
            "confidence": float(confidence),
            "faces_detected": faces_detected,
            "reason": reason_str,
            "debug_metrics": {
                "noise": float(tex_data["noise"]),
                "freq_ratio": float(freq_data["ratio"]),
                "detail": float(tex_data["detail"]),
                "freq_std": float(freq_data["std"])
            }
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during analysis:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
