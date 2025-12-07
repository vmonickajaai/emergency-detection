import os
import shutil

# -------------------------------
# UPDATE THESE PATHS
# -------------------------------
TRAIN_IMG_DIR = r"dataset/images/train"
TRAIN_LBL_DIR = r"dataset/labels/train"

VAL_IMG_DIR = r"dataset/images/val"
VAL_LBL_DIR = r"dataset/labels/val"

# -------------------------------
# CLASS NAME → ID MAPPING
# Modify only if needed
# -------------------------------
keyword_map = {
    "fire": 0,
    "accident": 1,
    "weapons": 2,
    "theif": 3,
    "fall": 4,
}

def detect_class(filename):
    name = filename.lower()
    for key, cid in keyword_map.items():
        if key in name:
            return cid
    return None


def fix_labels(IMG_DIR, LBL_DIR):
    print(f"\nProcessing: {LBL_DIR}")
    
    backup_dir = os.path.join(LBL_DIR, "backup_before_fix")
    os.makedirs(backup_dir, exist_ok=True)

    for fname in os.listdir(LBL_DIR):
        if not fname.endswith(".txt"):
            continue

        base = fname.replace(".txt", "")

        # find the matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(IMG_DIR, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print("⚠ No matching image for:", fname)
            continue

        cls = detect_class(os.path.basename(img_path))
        if cls is None:
            print("⚠ Could not detect class for:", img_path)
            continue

        # Read original label file
        lbl_file = os.path.join(LBL_DIR, fname)
        with open(lbl_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        new_lines = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 5:
                new_lines.append(f"{cls} " + " ".join(parts[1:5]))

        # backup original file
        shutil.copy2(lbl_file, os.path.join(backup_dir, fname))

        # write fixed file
        with open(lbl_file, "w") as f:
            f.write("\n".join(new_lines))

    print(f"✔ Done fixing {LBL_DIR}. Backup saved in:", backup_dir)



# -------------------------------
# RUN FIXING FOR BOTH TRAIN + VAL
# -------------------------------
fix_labels(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
fix_labels(VAL_IMG_DIR, VAL_LBL_DIR)

print("\n🎉 ALL LABELS FIXED SUCCESSFULLY!")
