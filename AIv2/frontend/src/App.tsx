import React, { useState } from 'react';
import axios from 'axios';
import { Upload, ShieldCheck, ShieldAlert, Loader2, Image as ImageIcon } from 'lucide-react';

interface AnalysisResult {
  success: boolean;
  prediction: string;
  confidence: number;
  faces_detected?: number;
  message?: string;
  reason?: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err: any) {
      console.error('Upload error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'An error occurred during analysis';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center py-12 px-4">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-bold mb-2 flex items-center justify-center gap-3">
          <ShieldCheck className="text-blue-500 w-10 h-10" />
          DeepGuard AI
        </h1>
        <p className="text-slate-400 max-w-md">
          Advanced Deepfake Detection System. Upload an image to verify its authenticity.
        </p>
      </header>

      <main className="w-full max-w-2xl bg-slate-800 rounded-2xl shadow-2xl p-8 border border-slate-700">
        <div className="flex flex-col items-center gap-6">
          {/* Upload Area */}
          <div 
            className={`w-full h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all cursor-pointer
              ${preview ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-slate-500 hover:bg-slate-700/50'}`}
            onClick={() => document.getElementById('fileInput')?.click()}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="h-full w-full object-contain p-2 rounded-xl" />
            ) : (
              <>
                <Upload className="w-12 h-12 text-slate-500 mb-4" />
                <p className="text-slate-400">Click to upload or drag and drop</p>
                <p className="text-slate-500 text-sm mt-1">PNG, JPG or JPEG</p>
              </>
            )}
            <input 
              id="fileInput"
              type="file" 
              className="hidden" 
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>

          {/* Action Button */}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className={`w-full py-4 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 transition-all
              ${!file || loading 
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-500 active:scale-[0.98]'}`}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" />
                Analyzing Media...
              </>
            ) : (
              'Verify Authenticity'
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="w-full p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-500 text-center">
              {error}
            </div>
          )}

          {/* Result Section */}
          {result && (
            <div className={`w-full p-6 rounded-xl border animate-in fade-in slide-in-from-bottom-4 duration-500
              ${result.prediction === 'Real' 
                ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' 
                : 'bg-rose-500/10 border-rose-500/50 text-rose-400'}`}>
              
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {result.prediction === 'Real' ? (
                    <ShieldCheck className="w-8 h-8" />
                  ) : (
                    <ShieldAlert className="w-8 h-8" />
                  )}
                  <h2 className="text-2xl font-bold uppercase tracking-wider">
                    {result.prediction} detected
                  </h2>
                </div>
                <div className="text-right">
                  <p className="text-sm opacity-75">Confidence</p>
                  <p className="text-xl font-mono font-bold">
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {result.success ? (
                <div className="space-y-2 text-sm">
                  <p className="flex justify-between">
                    <span>Faces Detected:</span>
                    <span className="font-semibold">{result.faces_detected}</span>
                  </p>
                  {result.reason && (
                    <p className="flex justify-between gap-4">
                      <span>Analysis Detail:</span>
                      <span className="font-semibold text-right">{result.reason}</span>
                    </p>
                  )}
                  <p className="opacity-80 italic mt-4">
                    {result.prediction === 'Real' 
                      ? "Analysis suggests this content is authentic and hasn't been significantly manipulated by AI."
                      : "Warning: High probability of AI generation or manipulation detected in this media."}
                  </p>
                </div>
              ) : (
                <p className="text-center">{result.message}</p>
              )}
            </div>
          )}
        </div>
      </main>

      <footer className="mt-12 text-slate-500 text-sm">
        Built with DeepGuard AI Technology &bull; 2026
      </footer>
    </div>
  );
}

export default App;