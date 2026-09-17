import json
import os
import sys
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

# Add the parent directory to sys.path so we can import the core logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive import compute_frame_energy, decode_adaptive, encode_adaptive
from core.audio_io import UnreadableAudio, read_wav
from core.decoder import decode_message
from core.encoder import calculate_capacity, encode_message
from core.steganalysis import run_full_steganalysis

# Uploads are read fully into memory before being written to disk, so an
# unbounded body size is a trivial memory-exhaustion vector. 200 MB comfortably
# covers the largest file in the research dataset (17 MB).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="CovertWave API")

# Wildcard origins with credentials is rejected by browsers and is not a
# meaningful policy; the frontend is served from this same origin, so no
# cross-origin access is needed at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class NumpyEncoder(json.JSONEncoder):
    """Serialises the numpy scalars the steganalysis battery returns."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def jsonable(payload):
    return json.loads(json.dumps(payload, cls=NumpyEncoder))


async def save_upload(file: UploadFile) -> str:
    """
    Validates and spools an uploaded WAV to a temp file, returning its path.
    Raises HTTPException on anything unacceptable.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(content)
        return tmp.name


def discard(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def validate_lsb(lsb_bits: int) -> int:
    if lsb_bits not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="lsb_bits must be 1, 2 or 4.")
    return lsb_bits


@app.post("/api/capacity")
async def get_capacity(file: UploadFile = File(...), lsb_bits: int = Form(1)):
    validate_lsb(lsb_bits)
    tmp_path = await save_upload(file)
    try:
        return JSONResponse(content=jsonable(calculate_capacity(tmp_path, lsb_bits=lsb_bits)))
    except (UnreadableAudio, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        discard(tmp_path)


@app.post("/api/encode")
async def encode(
    file: UploadFile = File(...),
    message: str = Form(...),
    password: str = Form(...),
    lsb_bits: int = Form(1),
    algorithm: str = Form("standard"),
):
    validate_lsb(lsb_bits)
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    tmp_in = await save_upload(file)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix="_stego.wav").name

    try:
        if algorithm == "adaptive":
            encode_adaptive(tmp_in, tmp_out, message, password, lsb_bits=lsb_bits)
        else:
            encode_message(tmp_in, tmp_out, message, password, lsb_bits=lsb_bits)

        safe_name = os.path.basename(file.filename or "audio.wav")
        # The stego file must outlive this handler so FileResponse can stream it;
        # BackgroundTask deletes it once the response has been sent. The previous
        # version simply never deleted it, leaking one WAV per encode request.
        return FileResponse(
            tmp_out,
            media_type="audio/wav",
            filename=f"stego_{safe_name}",
            background=BackgroundTask(discard, tmp_out),
        )
    except (UnreadableAudio, ValueError) as exc:
        discard(tmp_out)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        discard(tmp_out)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        discard(tmp_in)


@app.post("/api/decode")
async def decode(
    file: UploadFile = File(...),
    password: str = Form(...),
    lsb_bits: int = Form(1),
    algorithm: str = Form("standard"),
):
    validate_lsb(lsb_bits)
    tmp_path = await save_upload(file)

    try:
        if algorithm == "adaptive":
            message, positions, _ = decode_adaptive(tmp_path, password, lsb_bits=lsb_bits)
        else:
            message, positions, _ = decode_message(tmp_path, password, lsb_bits=lsb_bits)

        _, data = read_wav(tmp_path)
        if data.ndim > 1:
            data = data[:, 0]

        frame_size = max(1024, len(data) // 1000)
        energy = compute_frame_energy(data, frame_size=frame_size)
        bins = np.arange(len(energy) + 1) * frame_size
        density, _ = np.histogram(positions, bins=bins)

        return JSONResponse(content=jsonable({
            "message": message,
            "energy_profile": energy.tolist(),
            "payload_density": density.tolist(),
            "positions_used": len(positions),
            "bits_embedded": len(positions) * lsb_bits,
        }))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decoding failed: {exc}")
    finally:
        discard(tmp_path)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp_path = await save_upload(file)
    try:
        return JSONResponse(content=jsonable(run_full_steganalysis(tmp_path)))
    except (UnreadableAudio, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        discard(tmp_path)


# Mount the frontend directory to serve the UI
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Bind to loopback by default; set COVERTWAVE_HOST=0.0.0.0 to expose it
    # deliberately. Autoreload is a development convenience, not a default.
    uvicorn.run(
        "main:app",
        host=os.environ.get("COVERTWAVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("COVERTWAVE_PORT", "8000")),
        reload=os.environ.get("COVERTWAVE_RELOAD", "").lower() in ("1", "true", "yes"),
    )
