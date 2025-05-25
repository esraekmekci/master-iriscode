import os
import re
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
from itertools import product
import random
import math

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
output_path = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_code.png")
output_mask_path = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_codemask.png")
threshold = 0.32

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd_hdexe(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

def load_binary_image(path):
    return (np.array(Image.open(path).convert('L')) > 127).astype(np.uint8)

def save_binary_image(array, path):
    img = Image.fromarray((array * 255).astype(np.uint8))
    print(f"Saving binary image to {path}")
    img.save(path)

def load_all_templates(dataset_root, eye):  # eye = 'L' or 'R'
    templates = []

    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        if 'L' in subdirs or 'R' in subdirs:
            # CASIA format: subject/L/*.png
            eye_path = os.path.join(subject_path, eye)
            if not os.path.isdir(eye_path):
                continue
            for file in os.listdir(eye_path):
                if file.endswith("_code.png") and eye in file:
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(eye_path, base + "_code.png")
                    mask_path = os.path.join(eye_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base))

        else:
            # IITD format: subject/*.png
            for file in os.listdir(subject_path):
                if file.endswith("_code.png") and f"_{eye}_" in file:
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(subject_path, base + "_code.png")
                    mask_path = os.path.join(subject_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base))

    return templates


def evaluate_matches(code_path, mask_path, templates):
    count = 0
    for c, m in templates:
        hd = calculate_hd_hdexe(code_path, c, mask_path, m)
        if hd is not None and hd <= threshold:
            count += 1
    return count


def evaluate_matches_numpy(master_code, master_mask, templates):
    count = 0
    for i, (code, mask) in enumerate(templates):
        hd = calculate_hd_numpy(master_code, code, master_mask, mask)
        if hd is not None and hd <= threshold:
            count += 1
    return count


def calculate_hd_numpy(code1, code2, mask1, mask2):
    joint_mask = np.logical_and(mask1, mask2)
    if np.sum(joint_mask) == 0:
        return None
    diff = np.logical_xor(code1, code2)
    hd = np.sum(np.logical_and(diff, joint_mask)) / np.sum(joint_mask)
    return hd

def generate_initial_master_mode(codes, masks):
    H, W = codes[0].shape
    master = np.zeros((H, W), dtype=np.uint8)
    master_mask = np.ones((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            valid = [codes[k][i, j] for k in range(len(codes)) if masks[k][i, j] == 1]
            if len(valid) == 0:
                master_mask[i, j] = 0
            else:
                master[i, j] = 1 if sum(valid) > len(valid) / 2 else 0
    return master, master_mask

def greedy_bitwise_optimization(master, master_mask, templates):
    print("\n🔁 Improved greedy optimization starting (in-memory)...")
    H, W = master.shape

    current_score = evaluate_matches_numpy(master, master_mask, templates)
    indices = [(i, j) for i, j in product(range(H), range(W)) if master_mask[i, j] == 1]

    for i, j in tqdm(indices, desc="Optimizing bits", unit="px"):
        original_bit = master[i, j]
        master[i, j] = 1 - original_bit  # Flip

        new_score = evaluate_matches_numpy(master, master_mask, templates)

        if new_score >= current_score:
            current_score = new_score
        else:
            master[i, j] = original_bit  # Revert

    return master


def simulated_annealing_optimization(master, master_mask, templates,
                                     initial_temp=20.0, final_temp=0.000001, alpha=0.995, max_iter=100000):
    print("\n🔥 Simulated annealing optimization starting...")

    H, W = master.shape
    current_code = master.copy()
    current_score = evaluate_matches_numpy(current_code, master_mask, templates)

    temp = initial_temp
    history = [current_score]

    for iteration in tqdm(range(max_iter), desc="SA optimizing", unit="step"):
        # Rastgele geçerli bir piksel seç
        while True:
            i, j = random.randint(0, H-1), random.randint(0, W-1)
            if master_mask[i, j] == 1:
                break

        # Bit flip
        new_code = current_code.copy()
        new_code[i, j] = 1 - new_code[i, j]

        # Skor hesapla
        new_score = evaluate_matches_numpy(new_code, master_mask, templates)
        delta = new_score - current_score

        if delta >= 0:
            # İyileşme → Kabul et
            current_code = new_code
            current_score = new_score
        else:
            # Kötüleşme → Olasılıkla kabul et
            acceptance_prob = math.exp(delta / temp)
            if random.random() < acceptance_prob:
                current_code = new_code
                current_score = new_score

        history.append(current_score)

        # Sıcaklığı düşür
        temp = temp * alpha
        if temp < final_temp:
            break

    print(f"\n✅ Final Score (Simulated Annealing): {current_score} matches")
    return current_code


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['L', 'R']:
        print("❌ specify the eye: python generate_master_greedy_bitwise_eye.py L or python generate_master_greedy_bitwise_eye.py R")
        return

    eye = sys.argv[1]
    print(f"👁️ Processing only eye: {eye}")

    output_dir = os.path.join(dataset_root, f"master_{eye}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_code.png")
    output_mask_path = os.path.join(output_dir, "master_codemask.png")

    print("📂 Loading templates...")
    templates = load_all_templates(dataset_root, eye)
    print(f"✅ Loaded {len(templates)} templates for eye '{eye}'.")

    print("🧠 Loading iriscode arrays...")
    codes = [load_binary_image(c) for c, m, b in templates]
    masks = [load_binary_image(m) for c, m, b in templates]
    paired_templates = list(zip(codes, masks))

    print("📥 Creating initial master (mode fusion)...")
    master, master_mask = generate_initial_master_mode(codes, masks)
    save_binary_image(master_mask, output_mask_path)

    # 🔁 TEST: kodlar ters olabilir mi? kontrol et
    flipped_codes = [1 - code for code in codes]
    flipped_master, flipped_mask = generate_initial_master_mode(flipped_codes, masks)
    flipped_match = evaluate_matches_numpy(flipped_master, flipped_mask, paired_templates)
    print(f"🔁 TEST: flipped mode-fusion match score = {flipped_match}")

    print("🔧 Optimizing master template...")
    master = greedy_bitwise_optimization(master, master_mask, paired_templates)
    # master = simulated_annealing_optimization(master, master_mask, paired_templates)

    save_binary_image(master, output_path)
    print(f"\n💾 Master code saved: {output_path}")

    final_matches = evaluate_matches(master, master_mask, paired_templates)
    print(f"\n🏁 FINAL RESULT: matched {final_matches}/{len(templates)} templates (HD ≤ {threshold})")

if __name__ == "__main__":
    main()