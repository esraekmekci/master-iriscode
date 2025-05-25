import os
import re
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve

tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")

def run_hd(code1, code2, mask1, mask2):
    import subprocess
    cmd = [
        os.path.join(tool_path, "hd.exe"),
        "-i", os.path.abspath(code1), os.path.abspath(code2),
        "-m", os.path.abspath(mask1), os.path.abspath(mask2)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

def extract_id_name(path):
    # örnek: '001_01_L_code.png' → ('001', 'L')
    name = os.path.basename(path)
    parts = name.split('_')
    return parts[0], parts[2] if len(parts) >= 3 else None

def load_all_templates(dataset_root):
    templates = []  # list of (code_path, mask_path, base_name)

    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        if 'L' in subdirs or 'R' in subdirs:
            # CASIA/USIT format (with L and R folders)
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if not os.path.isdir(eye_path):
                    continue
                for file in os.listdir(eye_path):
                    if file.endswith("_code.png"):
                        base = file.replace("_code.png", "")
                        code_path = os.path.join(eye_path, base + "_code.png")
                        mask_path = os.path.join(eye_path, base + "_codemask.png")
                        if os.path.exists(code_path) and os.path.exists(mask_path):
                            templates.append((code_path, mask_path, base))
        else:
            # IITD format (flat files in subject folder)
            for file in os.listdir(subject_path):
                if file.endswith("_code.png"):
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(subject_path, base + "_code.png")
                    mask_path = os.path.join(subject_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base))
    return templates


def generate_labels_and_scores(templates):
    scores = []
    labels = []
    for (c1, m1, base1), (c2, m2, base2) in combinations(templates, 2):
        id1, eye1 = extract_id_name(base1)
        id2, eye2 = extract_id_name(base2)
        if eye1 != eye2:
            continue  # left vs right karşılaştırma yapma
        hd = run_hd(c1, c2, m1, m2)
        if hd is not None:
            label = 1 if id1 == id2 else 0
            labels.append(label)
            scores.append(hd)
    return np.array(labels), np.array(scores)


def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, -scores)  # -scores çünkü HD küçük olan daha iyi
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f"ROC Curve (EER ≈ {eer:.3f}, Optimum Threshold ≈ {abs(eer_threshold):.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"\n📌 Optimum HD Threshold ≈ {abs(eer_threshold):.4f} at Equal Error Rate ≈ {eer:.4f}")

def main():
    print("📂 Collecting iris templates...")
    templates = load_all_templates(dataset_root)
    print(f"✅ Found {len(templates)} templates.")

    print("⚙️ Generating genuine/impostor comparisons...")
    labels, scores = generate_labels_and_scores(templates)
    print(f"🔍 Total comparisons: {len(scores)}")

    print("📈 Plotting ROC and finding EER...")
    plot_roc(labels, scores)

if __name__ == "__main__":
    main()
