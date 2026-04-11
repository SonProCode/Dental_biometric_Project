# Dental Biometric Recognition Web App

## Quick Start

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Run the server (from project root)
python app/main.py

# 3. Open in browser
#    http://localhost:8000
```

## Project Structure

```
DentalWebApp/
├── app/
│   ├── main.py            # FastAPI app + all API routes
│   ├── yolo_utils.py      # Phase 1: YOLO segmentation + crop_polygon
│   ├── feature_utils.py   # Phase 2: ResNet18 features + cosine matching
│   ├── database_utils.py  # Feature DB CRUD (pickle)
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── model/
│   ├── yolo_best.pt       ← YOLO segmentation model
│   ├── phase2_model.pth   ← ResNet18 feature model
│   └── feature_db.pkl     ← auto-created on first person added
├── data/
│   └── database_images/   ← per-person X-ray images
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/list_persons` | List all registered persons |
| `POST` | `/recognize` | Recognize from X-ray upload |
| `POST` | `/add_person` | Register a new person |
| `DELETE` | `/delete_person?name=...` | Remove a person |
