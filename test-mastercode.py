import os
import re
import subprocess
from collections import defaultdict
import numpy as np
from PIL import Image

# === CONFIGURATION ===
tool_path       = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root    = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_dir      = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_casia_L")
master_code_path= os.path.join(output_dir, "master_code.png")
master_mask_path= os.path.join(output_dir, "master_codemask.png")
threshold       = 0.3727

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

def load_templates(dataset_root, eye=None):
    """
    dataset_root: kök klasör
    eye: "L", "R" veya None
    """
    templates = []
    subject_templates = defaultdict(list)
    
    for subject in sorted(os.listdir(dataset_root)):
        path_subj = os.path.join(dataset_root, subject)
        if not os.path.isdir(path_subj):
            continue

        # Altında L/R klasörü var mı diye bak
        subdirs = [d for d in os.listdir(path_subj) if os.path.isdir(os.path.join(path_subj, d))]
        is_casia = 'L' in subdirs or 'R' in subdirs

        if is_casia:
            # CASIA tarzı: L ve R altklasörleri
            targets = [eye] if eye in ('L','R') else ['L','R']
            for e in targets:
                dir_eye = os.path.join(path_subj, e)
                if not os.path.isdir(dir_eye):
                    continue
                for f in os.listdir(dir_eye):
                    if f.endswith("_code.png"):
                        base = f[:-9]  # remove _code.png
                        code_path = os.path.join(dir_eye, base + "_code.png")
                        mask_path = os.path.join(dir_eye, base + "_codemask.png")
                        if os.path.exists(mask_path):
                            templates.append((subject, base, code_path, mask_path))
                            subject_templates[subject].append((code_path, mask_path, base))
        else:
            # IITD tarzı: tek klasör içinde kod dosyaları, dosya adı _L veya _R içeriyorsa filtrele
            for f in os.listdir(path_subj):
                if not f.endswith("_code.png"):
                    continue
                if eye is None or f.endswith(f"_{eye}_code.png") or f.split('_')[-2] == eye:
                    base = f[:-9]
                    code_path = os.path.join(path_subj, base + "_code.png")
                    mask_path = os.path.join(path_subj, base + "_codemask.png")
                    if os.path.exists(mask_path):
                        templates.append((subject, base, code_path, mask_path))
                        subject_templates[subject].append((code_path, mask_path, base))

    return templates, subject_templates

def main(eye=None):
    print("📥 Loading templates...")
    templates, subject_templates = load_templates(dataset_root, eye=eye)
    print(f"✅ Loaded {len(templates)} templates (eye={eye})")

    matched = []
    matched_subjects = set()

    for subject, base, code_path, mask_path in templates:
        hd = calculate_hd(master_code_path, code_path, master_mask_path, mask_path)
        if hd is not None and 0.0 < hd <= threshold:
            matched.append((subject, base, hd))
            matched_subjects.add(subject)

    print("\n🎯 RESULT SUMMARY")
    print(f"✔️  Matched templates: {len(matched)} / {len(templates)}")
    print(f"✔️  Unique matched subjects: {len(matched_subjects)}\n")

    print("📋 Matched templates:")
    for subject, base, hd in matched:
        print(f" - {subject}/{base}: HD = {hd:.4f}")

if __name__ == "__main__":
    # Örneğin sadece sol göz için:
    # main(eye="L")
    # Sadece sağ göz için:
    # main(eye="R")
    # Her iki göz:
    main(eye="L")
