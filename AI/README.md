# DeepGuard AI - Deepfake Detection System

DeepGuard AI is a prototype system designed to detect AI-generated or manipulated images using Computer Vision and Neural Networks.

## Project Structure

- `backend/`: FastAPI server for image processing and AI inference.
- `frontend/`: React + Vite frontend for user interaction.

## Prerequisites

- Python 3.9+
- Node.js & npm

## Getting Started

### 1. Start the Backend

Navigate to the `backend` directory, install dependencies, and run the server:

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend will be available at `http://localhost:8000`.

### 2. Start the Frontend

Navigate to the `frontend` directory, install dependencies, and run the development server:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Features

- **Face Detection:** Automatically identifies face regions in uploaded media.
- **Deepfake Analysis:** Processes image frames through a neural network to identify manipulation artifacts.
- **Confidence Scoring:** Provides a percentage-based confidence level for the "Real" vs "Fake" classification.
- **Modern UI:** Responsive dashboard built with React and Tailwind CSS.

## AI Model Note
This prototype uses a simulated inference engine for demonstration. In a production environment, replace the model architecture in `backend/main.py` with a pre-trained weight file (e.g., EfficientNet or Vision Transformer) trained on datasets like FaceForensics++.
