"""
Feature database CRUD utilities.
db structure: {person_name: [np.ndarray, ...]}
"""
import pickle
import shutil
from pathlib import Path

FEATURE_DB_V1_PATH = Path(__file__).parent.parent / "model" / "feature_db.pkl"
FEATURE_DB_V2_PATH = Path(__file__).parent.parent / "model" / "feature_db_v2.pkl"
FEATURE_DB_V3_PATH = Path(__file__).parent.parent / "model" / "feature_db_v3.pkl"
DATABASE_IMAGES_DIR = Path(__file__).parent.parent / "data" / "database_images"


def get_db_path(version: int = 3) -> Path:
    if version == 3:
        return FEATURE_DB_V3_PATH
    return FEATURE_DB_V2_PATH if version == 2 else FEATURE_DB_V1_PATH


def load_db(version: int = 3) -> dict:
    path = get_db_path(version)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(db: dict, version: int = 3) -> None:
    path = get_db_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)


def list_persons(db: dict) -> list[str]:
    return sorted(db.keys())


def add_person(name: str, features_data, db: dict, version: int = 3) -> dict:
    """
    Add `features_data` (list or dict) for `name` to `db`.
    Raises ValueError if name already exists.
    Returns updated db.
    """
    if name in db:
        raise ValueError(f"Person '{name}' already exists in the database.")
    db[name] = features_data
    save_db(db, version)
    return db


def append_features(name: str, features_data, db: dict, version: int = 3) -> dict:
    """
    Append additional features to an existing person entry (used for multi-image add).
    If version is 3, features_data is expected to be a dict of {tooth_id: [features]}.
    """
    if name not in db:
        db[name] = {} if version >= 3 else []
        
    if version >= 3:
        for t_id, feats in features_data.items():
            if t_id not in db[name]:
                db[name][t_id] = []
            db[name][t_id].extend(feats)
    else:
        db[name].extend(features_data)
        
    save_db(db, version)
    return db


def delete_person(name: str, db: dict, version: int = 3) -> dict:
    """
    Remove person from db and delete their image folder.
    Raises KeyError if name not found.
    Returns updated db.
    """
    if name not in db:
        raise KeyError(f"Person '{name}' not found in the database.")
    del db[name]
    save_db(db, version)

    # Note: deleting the image folder might affect other DB versions if they exist.
    # But usually, they should be in sync.
    person_dir = DATABASE_IMAGES_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)

    return db


def save_database_image(person_name: str, image_bytes: bytes, filename: str) -> Path:
    """Save an uploaded X-ray image to data/database_images/{person}/{filename}."""
    person_dir = DATABASE_IMAGES_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    dest = person_dir / filename
    dest.write_bytes(image_bytes)
    return dest
