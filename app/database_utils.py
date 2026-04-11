"""
Feature database CRUD utilities.
db structure: {person_name: [np.ndarray, ...]}
"""
import pickle
import shutil
from pathlib import Path

FEATURE_DB_PATH = Path(__file__).parent.parent / "model" / "feature_db.pkl"
DATABASE_IMAGES_DIR = Path(__file__).parent.parent / "data" / "database_images"


def load_db() -> dict:
    if not FEATURE_DB_PATH.exists():
        return {}
    with open(FEATURE_DB_PATH, "rb") as f:
        return pickle.load(f)


def save_db(db: dict) -> None:
    FEATURE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_DB_PATH, "wb") as f:
        pickle.dump(db, f)


def list_persons(db: dict) -> list[str]:
    return sorted(db.keys())


def add_person(name: str, features_list: list, db: dict) -> dict:
    """
    Add `features_list` (list of np.ndarray) for `name` to `db`.
    Raises ValueError if name already exists.
    Returns updated db.
    """
    if name in db:
        raise ValueError(f"Person '{name}' already exists in the database.")
    db[name] = features_list
    save_db(db)
    return db


def append_features(name: str, features_list: list, db: dict) -> dict:
    """
    Append additional features to an existing person entry (used for multi-image add).
    """
    if name not in db:
        db[name] = []
    db[name].extend(features_list)
    save_db(db)
    return db


def delete_person(name: str, db: dict) -> dict:
    """
    Remove person from db and delete their image folder.
    Raises KeyError if name not found.
    Returns updated db.
    """
    if name not in db:
        raise KeyError(f"Person '{name}' not found in the database.")
    del db[name]
    save_db(db)

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
