"""
Phase 2 – Feature Extraction (ResNet18) + Cosine Matching.
Includes Model V2 support.
"""
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from pathlib import Path

PHASE2_MODEL_PATH = Path(__file__).parent.parent / "model" / "phase2_model.pth"
PHASE2_MODEL_V2_PATH = Path(__file__).parent.parent / "model" / "phase2_model_v2_best.pth"

device = torch.device("cpu")

_feature_model_v1 = None
_feature_model_v2 = None

# V1 Transform
_transform_v1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# V2 Transform — must exactly match train
_transform_v2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ToothNet(nn.Module):
    """Model V2 architecture."""
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # Embedding: Linear + BatchNorm, NO ReLU
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        feat = self.embedding(x)
        # L2-normalize — same as train
        feat = F.normalize(feat, dim=1)
        return feat


def get_feature_model_v1() -> nn.Module:
    global _feature_model_v1
    if _feature_model_v1 is None:
        model = resnet18(weights=None)
        model.fc = nn.Identity()
        if PHASE2_MODEL_PATH.exists():
            state = torch.load(str(PHASE2_MODEL_PATH), map_location=device)
            model.load_state_dict(state, strict=False)
        model.eval()
        _feature_model_v1 = model.to(device)
    return _feature_model_v1


def get_feature_model_v2() -> nn.Module:
    global _feature_model_v2
    if _feature_model_v2 is None:
        model = ToothNet()
        if PHASE2_MODEL_V2_PATH.exists():
            ckpt = torch.load(str(PHASE2_MODEL_V2_PATH), map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt, strict=False)
        model.eval()
        _feature_model_v2 = model.to(device)
    return _feature_model_v2


def extract_feature(tooth_img: np.ndarray, version: int = 2) -> np.ndarray:
    """
    Given a (H, W, 3) BGR numpy tooth image, return a normalized embedding.
    version 1: 512-d
    version 2: 256-d (default)
    """
    rgb = cv2.cvtColor(tooth_img, cv2.COLOR_BGR2RGB)
    
    if version == 1:
        tensor = _transform_v1(rgb).unsqueeze(0).to(device)
        model = get_feature_model_v1()
        with torch.no_grad():
            feat = model(tensor).squeeze().cpu().numpy()
        # V1 manual normalization
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
    else:
        tensor = _transform_v2(rgb).unsqueeze(0).to(device)
        model = get_feature_model_v2()
        with torch.no_grad():
            feat = model(tensor).squeeze().cpu().numpy()
        # V2 model output is already normalized
        
    return feat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Both a and b are expected to be L2-normalized
    return float(np.dot(a, b))


def match_identity(
    query_features: list[np.ndarray],
    db: dict,
    top_k: int = 10
) -> list[dict]:
    """
    Compare query_features against every person in db.
    db structure: {person_name: [feat_vec, feat_vec, ...]}
    """
    if not db:
        return []

    ranked = []
    for person, stored_feats in db.items():
        if not stored_feats:
            continue
            
        db_matrix = np.stack(stored_feats)  # (N_db, dim)
        
        query_best_scores = []
        for qf in query_features:
            # dot product = cosine similarity because both are normalized
            sims = db_matrix @ qf
            best_sim = float(sims.max())
            query_best_scores.append(best_sim)
        
        if not query_best_scores:
            continue

        # Top-K Mean scoring
        top_k_scores = sorted(query_best_scores, reverse=True)[:top_k]
        avg_score = float(np.mean(top_k_scores))

        ranked.append({
            "name": person,
            "score": round(avg_score, 4),
            "num_matched": len(query_best_scores),
        })

    # Sort results
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
