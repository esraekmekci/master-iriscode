import os
import re
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
output_path = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_code.png")
output_mask_path = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/master_codemask.png")
threshold = 0.32

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def run_segmentation_and_feature_extraction(image_path, out_base):
    texture = out_base + "_texture.png"
    mask = out_base + "_mask.png"
    code = out_base + "_code.png"
    codemask = out_base + "_codemask.png"

    if not os.path.exists(code):
        run_command([os.path.join(tool_path, "caht.exe"), "-i", image_path, "-o", texture, "-m", mask, "-q"])
        run_command([os.path.join(tool_path, "qsw.exe"), "-i", texture, "-m", mask, codemask, "-o", code, "-q"])
    return code, codemask

def calculate_hd_hdexe(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    try:
        match = re.search(r"=\s*([0-9.]+)", result.stdout)
        if match:
            return float(match.group(1))
    except:
        return None
    return None

def load_binary_image(path):
    return np.array(Image.open(path).convert('1')).astype(np.uint8)

def save_binary_image(array, path):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)

def load_all_templates(dataset_root):
    templates = []  # list of (code_path, mask_path, base_name)
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        if 'L' in subdirs or 'R' in subdirs:
            # CASIA/USIT format
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if not os.path.isdir(eye_path):
                    continue
                for file in os.listdir(eye_path):
                    if file.endswith(".jpg") or file.endswith(".bmp") or file.endswith(".png"):
                        base = os.path.splitext(file)[0]
                        base_name = f"{base}"
                        img_path = os.path.join(eye_path, file)
                        out_base = os.path.join(eye_path, base)
                        code, codemask = run_segmentation_and_feature_extraction(img_path, out_base)
                        if os.path.exists(code) and os.path.exists(codemask):
                            templates.append((code, codemask, base_name))
        else:
            # IITD format
            for file in os.listdir(subject_path):
                if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png"):
                    base = os.path.splitext(file)[0]
                    base_name = f"{subject}_{base}"
                    img_path = os.path.join(subject_path, file)
                    out_base = os.path.join(subject_path, base_name)
                    code, codemask = run_segmentation_and_feature_extraction(img_path, out_base)
                    if os.path.exists(code) and os.path.exists(codemask):
                        templates.append((code, codemask, base_name))
    return templates

def generate_greedy_master_code(templates):
    print("\n🧠 Building master iriscode (greedy)...")
    sample_code = load_binary_image(templates[0][0])
    H, W = sample_code.shape
    master = np.zeros((H, W), dtype=np.uint8)
    master_mask = np.ones((H, W), dtype=np.uint8)

    all_codes = []
    all_masks = []
    for code_path, mask_path, _ in templates:
        all_codes.append(load_binary_image(code_path))
        all_masks.append(load_binary_image(mask_path))

    for i in tqdm(range(H)):
        for j in range(W):
            count_0 = count_1 = 0
            for k in range(len(all_codes)):
                if all_masks[k][i, j] == 0:
                    continue
                if all_codes[k][i, j] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            if count_0 == 0 and count_1 == 0:
                master_mask[i, j] = 0
            else:
                master[i, j] = 1 if count_1 > count_0 else 0
    return master, master_mask

def evaluate_with_hdexe(master_code_path, master_mask_path, templates):
    print("\n🔍 Evaluating master code with hd.exe...")
    match_count = 0
    total = len(templates)
    for code_path, mask_path, name in tqdm(templates):
        hd = calculate_hd_hdexe(master_code_path, code_path, master_mask_path, mask_path)
        if hd is not None and hd <= threshold:
            match_count += 1
    return match_count, total

def main():
    print("📂 Loading and preprocessing templates...")
    templates = load_all_templates(dataset_root)
    print(f"✅ Loaded {len(templates)} iris samples.")

    master_code, master_mask = generate_greedy_master_code(templates)

    save_binary_image(master_code, output_path)
    save_binary_image(master_mask, output_mask_path)
    print(f"\n💾 Master code saved to: {output_path}")
    print(f"💾 Master mask saved to: {output_mask_path}")

    matched, total = evaluate_with_hdexe(output_path, output_mask_path, templates)
    print(f"\n🏁 FINAL RESULT: matched {matched}/{total} templates (HD ≤ {threshold})")

if __name__ == "__main__":
    main()

