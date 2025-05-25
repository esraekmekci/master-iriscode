import os
import re
import subprocess
import numpy as np
from PIL import Image

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_dir = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_R")
master_code_path = os.path.join(output_dir, "master_code.png")
master_mask_path = os.path.join(output_dir, "master_codemask.png")
threshold = 0.3511

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

def load_templates(dataset_root):
    templates = []
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        if 'L' in subdirs or 'R' in subdirs:
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if not os.path.isdir(eye_path):
                    continue
                for file in os.listdir(eye_path):
                    if file.endswith("_code.png"):
                        base = file.replace("_code.png", "")
                        code_path = os.path.join(eye_path, base + "_code.png")
                        mask_path = os.path.join(eye_path, base + "_codemask.png")
                        if os.path.exists(mask_path):
                            templates.append((subject, base, code_path, mask_path))
        else:
            for file in os.listdir(subject_path):
                if file.endswith("_code.png"):
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(subject_path, base + "_code.png")
                    mask_path = os.path.join(subject_path, base + "_codemask.png")
                    if os.path.exists(mask_path):
                        templates.append((subject, base, code_path, mask_path))
    return templates

def main():
    print("📥 Loading templates...")
    templates = load_templates(dataset_root)
    print(f"✅ Loaded {len(templates)} templates")

    matched_templates = []
    matched_subjects = set()

    for subject, base, code_path, mask_path in templates:
        hd = calculate_hd(master_code_path, code_path, master_mask_path, mask_path)
        if hd is not None and 0.0 < hd <= threshold:
            matched_templates.append((subject, base, hd))
            matched_subjects.add(subject)

    print(f"\n🎯 RESULT SUMMARY")
    print(f"✔️  Matched templates: {len(matched_templates)} / {len(templates)}")
    print(f"✔️  Unique matched subjects: {len(matched_subjects)}")

    print("\n📋 Matched templates:")
    for subject, base, hd in matched_templates:
        print(f" - {subject}/{base}: HD = {hd:.4f}")

if __name__ == "__main__":
    main()
