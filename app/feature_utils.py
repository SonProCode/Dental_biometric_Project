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
    query_features: list[tuple[int, np.ndarray]],
    db: dict,
    top_k: int = 10,
    neighbor_range: int = 1,
    penalty_factor: float = 0.02,
    threshold: float = 0.75,
    debug: bool = False
) -> list[dict]:
    """
    Compare query_features against every person in db using top-k scoring logic with positional penalty.
    db structure: {person_name: {tooth_id: [feat_vec, feat_vec, ...]}}
    query_features structure: [(tooth_id, feat_vec), ...]
    """
    if not db:
        return []

    ranked = []
    
    # 1. Pre-validate query features
    valid_query_features = []
    for tooth_id, query_feat in query_features:
        if not np.isfinite(query_feat).all():
            if debug:
                print(f"tooth_{tooth_id} -> [SKIP] query feature has NaN/Inf")
            continue
            
        # Optional: Re-normalize query feature
        norm = np.linalg.norm(query_feat)
        if norm > 0:
            query_feat = query_feat / norm
            
        valid_query_features.append((tooth_id, query_feat))

    for person, db_data in db.items():
        if not db_data:
            continue
            
        person_matches = []
        for tooth_id, query_feat in valid_query_features:
            candidate_ids = range(
                max(0, tooth_id - neighbor_range),
                min(31, tooth_id + neighbor_range) + 1
            )
            
            best_score = -np.inf
            match_cid = -1
            
            for cid in candidate_ids:
                if cid not in db_data:
                    continue
                    
                db_feats = db_data[cid]
                if len(db_feats) == 0:
                    continue
                    
                db_matrix = np.stack(db_feats)
                # Check for NaN/Inf in DB vectors
                if not np.isfinite(db_matrix).all():
                    continue
                    
                if db_matrix.ndim == 1:
                    db_matrix = db_matrix.reshape(1, -1)
                    
                sims = db_matrix @ query_feat
                local_best = float(sims.max())
                
                # Penalty for positional offset
                penalty = penalty_factor * abs(cid - tooth_id)
                local_best -= penalty
                
                if local_best > best_score:
                    best_score = local_best
                    match_cid = cid
                    
            if best_score > -np.inf:
                person_matches.append(best_score)
                if debug:
                    print(f"tooth_{tooth_id} -> matched tooth_{match_cid} score={best_score:.4f}")
            else:
                if debug:
                    print(f"tooth_{tooth_id} -> no match")
        
        if not person_matches:
            continue

        # Top-K Mean scoring
        tk_actual = min(top_k, len(person_matches))
        top_k_scores = sorted(person_matches, reverse=True)[:tk_actual]
        mean_score = float(np.mean(top_k_scores))
        
        # Matched bonus
        matched_bonus = 0.002 * min(len(person_matches), 20)
        final_score = mean_score + matched_bonus
        
        # Clamp score cuối
        final_score = max(0.0, min(float(final_score), 1.0))

        ranked.append({
            "name": person,
            "score": final_score,
            "num_matched": len(person_matches),
            "top_k_used": tk_actual,
            "avg_top_k": mean_score,
            "matched_bonus": matched_bonus
        })

    # Sort results
    ranked.sort(key=lambda x: x["score"], reverse=True)
    
    # Thêm reject UNKNOWN ngay trong matcher
    if ranked and ranked[0]["score"] < threshold:
        best_unknown = ranked[0].copy()
        best_unknown["best_candidate"] = best_unknown["name"]
        best_unknown["name"] = "UNKNOWN"
        return [best_unknown]

    return ranked
