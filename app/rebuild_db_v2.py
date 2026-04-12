"""
Rebuild the feature database for Model V2 using stored images.
"""
import os
import sys
from pathlib import Path

# Add app dir to path
sys.path.insert(0, str(Path(__file__).parent))

from yolo_utils import segment_teeth
from feature_utils import extract_feature
from database_utils import DATABASE_IMAGES_DIR, save_db

def rebuild_v2():
    if not DATABASE_IMAGES_DIR.exists():
        print(f"Error: {DATABASE_IMAGES_DIR} does not exist.")
        return

    db_v2 = {}
    total_persons = 0
    total_features = 0

    print(f"Scanning {DATABASE_IMAGES_DIR}...")
    
    # Iterate through each person
    for person_dir in sorted(DATABASE_IMAGES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        print(f"\nProcessing [{person_name}]")
        
        person_features = []
        
        # Iterate through each image for this person
        image_files = [
            f for f in person_dir.iterdir() 
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        
        for img_path in image_files:
            print(f"  - Segmenting: {img_path.name}")
            try:
                # Phase 1: Segmentation
                teeth_imgs = segment_teeth(str(img_path))
                if not teeth_imgs:
                    print(f"    [WARN] No teeth detected in {img_path.name}")
                    continue
                
                # Phase 2: Feature Extraction (V2)
                for t_img in teeth_imgs:
                    feat = extract_feature(t_img, version=2)
                    person_features.append(feat)
                    total_features += 1
            except Exception as e:
                print(f"    [ERROR] {img_path.name}: {e}")
        
        if person_features:
            db_v2[person_name] = person_features
            total_persons += 1
            print(f"  → Extracted {len(person_features)} features")
        else:
            print(f"  [WARN] No features extracted for {person_name}")

    if db_v2:
        save_db(db_v2, version=2)
        print("\n" + "="*40)
        print("REBUILD COMPLETE")
        print(f"  Persons     : {total_persons}")
        print(f"  Total feats : {total_features}")
        print(f"  Saved to    : model/feature_db_v2.pkl")
    else:
        print("\nNo features were extracted. Database not saved.")

if __name__ == "__main__":
    rebuild_v2()
