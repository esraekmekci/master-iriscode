import os
import re
import subprocess
import numpy as np
from itertools import combinations
from PIL import Image

# CONFIGURATION
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_base = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-wolves")
threshold = 0.32
max_wolves = 10  # En çok eşleşen N wolf seçilsin

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def run_segmentation_and_feature_extraction(img_path):
    base = os.path.splitext(img_path)[0]
    texture = base + "_texture.png"
    mask = base + "_mask.png"
    code = base + "_code.png"
    codemask = base + "_codemask.png"

    if not os.path.exists(code):
        run_command([os.path.join(tool_path, "caht.exe"), "-i", img_path, "-o", texture, "-m", mask, "-q"])
        run_command([os.path.join(tool_path, "qsw.exe"), "-i", texture, "-m", mask, codemask, "-o", code, "-q"])

    try:
        code_array = np.array(Image.open(code).convert('1')).astype(np.uint8)
        if np.mean(code_array) in [0.0, 1.0]:
            print(f"[DISCARDED] {base}: all 0s or 1s")
            return None, None, None
    except:
        print(f"[ERROR] cannot load {code}")
        return None, None, None

    return os.path.abspath(code), os.path.abspath(codemask), os.path.basename(base)

def calculate_hd(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    try:
        match = re.search(r"=\s*([0-9.]+)", result.stdout)
        if match:
            val = float(match.group(1))
            return val if val > 0 else None
    except:
        return None
    return None

def load_binary_image(path):
    return np.array(Image.open(path).convert('1')).astype(np.uint8)

def save_binary_image(array, filename, output_dir):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(os.path.join(output_dir, filename))

def mix_codes_array(code1, code2, op):
    if op == 'and':
        return np.bitwise_and(code1, code2)
    elif op == 'or':
        return np.bitwise_or(code1, code2)
    elif op == 'xor':
        return np.bitwise_xor(code1, code2)
    else:
        raise ValueError("Invalid op")

def process_eye(all_entries, eye_label, output_dir):
    print(f"\n=== Processing eye: {eye_label} ===")
    os.makedirs(output_dir, exist_ok=True)

    # Filter entries for current eye
    eye_codes = [entry for entry in all_entries if entry[1] == eye_label]
    print(f"Total {eye_label} samples: {len(eye_codes)}")

    # --- WOLF SELECTION ---
    print(f"\n📈 Calculating wolf scores for {eye_label} eye...")
    wolf_score = {}
    for (s1, e1, b1, c1, m1), (s2, e2, b2, c2, m2) in combinations(eye_codes, 2):
        if s1 == s2:
            continue  # same subject
        hd = calculate_hd(c1, c2, m1, m2)
        if hd is not None and hd <= threshold:
            wolf_score[b1] = wolf_score.get(b1, 0) + 1
            wolf_score[b2] = wolf_score.get(b2, 0) + 1

    top_wolves = sorted(wolf_score.items(), key=lambda x: -x[1])[:max_wolves]
    wolf_names = [w[0] for w in top_wolves]

    print(f"\n🐺 Top {eye_label} wolves:")
    for name, score in top_wolves:
        print(f" - {name}: {score} matches")

    wolf_entries = [entry for entry in eye_codes if entry[2] in wolf_names]
    wolf_dict = {b: (c, m) for (_, _, b, c, m) in wolf_entries}
    code_arrays = {b: load_binary_image(c) for (_, _, b, c, _) in wolf_entries}

    print("\n🧪 Generating and testing all alpha-wolf combinations...")
    best_match_count = -1
    best_info = None

    for (b1, b2) in combinations(wolf_names, 2):
        for op in ['and', 'or', 'xor']:
            mixed_array = mix_codes_array(code_arrays[b1], code_arrays[b2], op)
            fname = f"{b1}_{b2}_{op}.png"
            save_binary_image(mixed_array, fname, output_dir)

            match_count = 0
            for (_, _, _, c, m) in eye_codes:
                hd = calculate_hd(os.path.join(output_dir, fname), c, m, m)
                if hd is not None and hd <= threshold:
                    match_count += 1

            print(f"[{eye_label.upper()} | {op.upper()}] {b1} + {b2} → {match_count} matches")
            if match_count > best_match_count:
                best_match_count = match_count
                best_info = (fname, match_count)

    if best_info:
        print(f"\n🏆 Best {eye_label} alpha-wolf: {best_info[0]} with {best_info[1]} matches")


def main():
    print("📂 Collecting all iris codes...")
    all_codes = []  # (subject, eye, base, code_path, mask_path)
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path): continue
        for eye in ['L', 'R']:
            eye_path = os.path.join(subject_path, eye)
            if not os.path.isdir(eye_path): continue
            for file in os.listdir(eye_path):
                if file.endswith(".jpg"):
                    img_path = os.path.join(eye_path, file)
                    code, codemask, base = run_segmentation_and_feature_extraction(img_path)
                    if code and codemask:
                        all_codes.append((subject, eye, base, code, codemask))

    print(f"\n✅ Total valid codes: {len(all_codes)}")

    process_eye(all_codes, 'L', os.path.join(output_base, "left"))
    process_eye(all_codes, 'R', os.path.join(output_base, "right"))

if __name__ == "__main__":
    main()
