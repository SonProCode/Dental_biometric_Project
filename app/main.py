"""
Main FastAPI application – Dental Biometric Recognition Demo.
Run: python app/main.py  (from project root)
"""
import sys
import os
import base64
import tempfile
import uuid
import cv2
import numpy as np
import uvicorn

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Add app dir to path so sibling imports work
sys.path.insert(0, str(Path(__file__).parent))

from yolo_utils import segment_teeth
from feature_utils import extract_feature, match_identity
from database_utils import (
    load_db, save_db, list_persons,
    add_person, delete_person, save_database_image,
)

# ── App setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
app = FastAPI(title="Dental Biometric Recognition")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Helper ─────────────────────────────────────────────────────────────────
def ndarray_to_base64(img: np.ndarray) -> str:
    """Convert BGR numpy image to base64 PNG string for JSON transport."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def save_upload_to_tmp(upload: UploadFile) -> str:
    """Save an UploadFile to a temp path and return the path string."""
    suffix = Path(upload.filename).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/list_persons")
async def api_list_persons():
    db = load_db()
    return {"persons": list_persons(db)}


@app.post("/recognize")
async def api_recognize(file: UploadFile = File(...)):
    # --- save upload
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    tmp_path = save_upload_to_tmp(file)
    try:
        # Phase 1 – segmentation
        teeth_imgs = segment_teeth(tmp_path)
        if not teeth_imgs:
            return JSONResponse({
                "success": False,
                "message": "No teeth detected in the image. Try a clearer X-ray.",
                "tooth_count": 0,
            })

        # Phase 2 – extract features
        query_features = [extract_feature(t, version=2) for t in teeth_imgs]
        num_teeth = len(query_features)

        # Matching
        db = load_db(version=2)
        if not db:
            return JSONResponse({
                "success": False,
                "message": "Database (V2) is empty. Please add a person or rebuild the database.",
                "tooth_count": num_teeth,
            })

        ranked = match_identity(query_features, db, top_k=10)

        # Apply threshold to handle open-set recognition (unknown person)
        RECOGNITION_THRESHOLD = 0.75  # Updated for V2
        MIN_TEETH_REQUIRED = 3
        
        if not ranked:
            result_identity = "unknown"
            top_score = 0
        else:
            best = ranked[0]
            top_score = best["score"]
            # Combine threshold and min teeth checks
            if top_score < RECOGNITION_THRESHOLD or num_teeth < MIN_TEETH_REQUIRED:
                result_identity = "unknown"
            else:
                result_identity = best["name"]

        # Debug logging for recognition results
        print(f"\n=== [DEBUG] Recognition (Teeth: {num_teeth}) ===")
        print(f"Result: {result_identity}")
        for i, r in enumerate(ranked[:5]):
            print(f"{i+1}. {r['name']} - Score: {r['score']} ({r['num_matched']} teeth)")
        print("==============================================\n")

        # Encode tooth crops to base64
        tooth_b64 = [ndarray_to_base64(t) for t in teeth_imgs]

        return {
            "success": True,
            "result": result_identity,
            "score": top_score,
            "tooth_count": num_teeth,
            "top3": ranked[:3],
            "tooth_images": tooth_b64,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/add_person")
async def api_add_person(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Person name cannot be empty.")

    db = load_db(version=2)
    if name in db:
        raise HTTPException(status_code=400, detail=f"'{name}' already exists in the database (V2).")

    all_features = []
    tooth_total = 0
    tmp_paths = []

    try:
        for upload in files:
            if not upload.content_type.startswith("image/"):
                continue
            tmp_path = save_upload_to_tmp(upload)
            tmp_paths.append(tmp_path)

            # Save a copy to database_images
            upload.file.seek(0)
            img_bytes = Path(tmp_path).read_bytes()
            safe_filename = f"{uuid.uuid4().hex}{Path(upload.filename).suffix or '.jpg'}"
            save_database_image(name, img_bytes, safe_filename)

            # Phase 1 + 2
            teeth = segment_teeth(tmp_path)
            tooth_total += len(teeth)
            for t in teeth:
                all_features.append(extract_feature(t, version=2))

        if not all_features:
            raise HTTPException(
                status_code=400,
                detail="No teeth detected in any of the uploaded images."
            )

        add_person(name, all_features, db, version=2)

    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)

    return {
        "success": True,
        "message": f"Added '{name}' with {len(all_features)} feature vectors from {tooth_total} teeth.",
    }


@app.delete("/delete_person")
async def api_delete_person(name: str):
    name = name.strip()
    db = load_db()
    try:
        delete_person(name, db)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"success": True, "message": f"Deleted '{name}' from the database."}


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
