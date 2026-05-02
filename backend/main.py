import sys
import os
import io
import json
import tempfile
import numpy as np
from scipy.io import wavfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add the parent directory to sys.path so we can import the core logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoder import encode_message, calculate_capacity
from core.decoder import decode_message
from core.adaptive import encode_adaptive, decode_adaptive, compute_frame_energy
from core.steganalysis_extended import run_full_steganalysis

app = FastAPI(title="CovertWave API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/capacity")
async def get_capacity(file: UploadFile = File(...), lsb_bits: int = Form(1)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap_info = calculate_capacity(tmp_path, lsb_bits=lsb_bits)
        return JSONResponse(content=cap_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/encode")
async def encode(
    file: UploadFile = File(...),
    message: str = Form(...),
    password: str = Form(...),
    lsb_bits: int = Form(1),
    algorithm: str = Form("standard")
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".wav", "_stego.wav")

    try:
        if algorithm == "adaptive":
            encode_adaptive(tmp_in_path, tmp_out_path, message, password, lsb_bits=lsb_bits)
        else:
            encode_message(tmp_in_path, tmp_out_path, message, password, lsb_bits=lsb_bits)
            
        return FileResponse(
            tmp_out_path, 
            media_type="audio/wav", 
            filename=f"stego_{file.filename}",
            background=None # we shouldn't delete immediately if FileResponse is reading it. Let's use a background task or just rely on OS temp cleanup.
        )
    except Exception as e:
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        # Note: tmp_out_path is removed in a background task ideally, but for now we let it be served.


@app.post("/api/decode")
async def decode(
    file: UploadFile = File(...),
    password: str = Form(...),
    lsb_bits: int = Form(1),
    algorithm: str = Form("standard")
):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if algorithm == "adaptive":
            message, positions, total_samples = decode_adaptive(tmp_path, password, lsb_bits=lsb_bits)
        else:
            message, positions, total_samples = decode_message(tmp_path, password, lsb_bits=lsb_bits)
            
        # Calculate Energy Profile & Payload Density for the UI (downsampled to ~1000 points)
        sample_rate, data = wavfile.read(tmp_path)
        if len(data.shape) > 1:
            data = data[:, 0]
        frame_size = max(1024, len(data) // 1000)
        
        energy = compute_frame_energy(data, frame_size=frame_size)
        num_frames = len(energy)
        bins = np.arange(num_frames + 1) * frame_size
        density, _ = np.histogram(positions, bins=bins)
        
        return JSONResponse(content={
            "message": message,
            "energy_profile": energy.tolist(),
            "payload_density": density.tolist()
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail="Decoding failed: " + str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = run_full_steganalysis(tmp_path)
        
        # We need to make sure results are JSON serializable. 
        # run_full_steganalysis returns standard python dicts but some values might be numpy types.
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
                
        json_data = json.loads(json.dumps(results, cls=NpEncoder))
        return JSONResponse(content=json_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Mount the frontend directory to serve the UI
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
