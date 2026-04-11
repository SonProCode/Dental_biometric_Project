"""
Phase 2 – Feature Extraction (ResNet18) + Cosine Matching.
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from pathlib import Path

PHASE2_MODEL_PATH = Path(__file__).parent.parent / "model" / "phase2_model.pth"

device = torch.device("cpu")

_feature_model = None

# Grayscale normalisation (as suggested by the model user)
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def get_feature_model() -> nn.Module:
    global _feature_model
    if _feature_model is None:
        model = resnet18(weights=None)
        model.fc = nn.Identity()
        state = torch.load(str(PHASE2_MODEL_PATH), map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        _feature_model = model.to(device)
    return _feature_model


def extract_feature(tooth_img: np.ndarray) -> np.ndarray:
    """
    Given a (H, W, 3) BGR numpy tooth image, return a 512-d L2-normalised embedding.
    """
    rgb = cv2.cvtColor(tooth_img, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0).to(device)

    model = get_feature_model()
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()

    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def match_identity(
    query_features: list[np.ndarray],
    db: dict
) -> list[dict]:
    """
    Compare query_features against every person in db.
    db structure: {person_name: [feat_vec, feat_vec, ...]}

    Logic:
    1. For each query tooth: find the BEST match score in the person's stored features.
    2. Collect all these best scores.
    3. Take Top-10 best scores and average them.
    """
    if not db:
        return []

    ranked = []
    for person, stored_feats in db.items():
        # Step 1 & 2: Best match score per query tooth
        query_best_scores = []
        for qf in query_features:
            best_sim = max([cosine_similarity(qf, sf) for sf in stored_feats], default=0)
            query_best_scores.append(best_sim)
        
        if not query_best_scores:
            continue

        # Step 3: Top-10 Mean
        top_k_scores = sorted(query_best_scores, reverse=True)[:10]
        avg_score = float(np.mean(top_k_scores))

        ranked.append({
            "name": person,
            "score": round(avg_score, 4),
            "num_matched": len(query_best_scores),
        })

    # Sort results
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
