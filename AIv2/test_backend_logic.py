import cv2
import numpy as np
import io
from PIL import Image
import os

def analyze_frequency_artifacts(img_gray):
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (center_w, center_h), radius, 0, -1)
    high_freq_energy = np.mean(magnitude_spectrum[mask == 1])
    total_energy = np.mean(magnitude_spectrum)
    ratio = high_freq_energy / (total_energy + 1e-6)
    return ratio

def analyze_noise_consistency(img_gray):
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_var

def test_logic():
    print("Creating dummy image...")
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    
    img_np = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Simulate finding a face (whole image as face)
    face_gray = gray[100:300, 100:300]
    
    print("Testing Frequency Analysis...")
    freq = analyze_frequency_artifacts(face_gray)
    print(f"Freq: {freq}")
    
    print("Testing Noise Analysis...")
    noise = analyze_noise_consistency(face_gray)
    print(f"Noise: {noise}")
    
    print("Logic test passed!")

if __name__ == "__main__":
    try:
        test_logic()
    except Exception as e:
        import traceback
        traceback.print_exc()
