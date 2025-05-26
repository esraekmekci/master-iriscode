import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random
import subprocess
import re
import tempfile
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-ml-iriscode")
threshold = 0.32  # Increased threshold based on your results

# === UTILITY FUNCTIONS ===
def load_binary_image(path):
    # Load as binary and ensure proper integer type to avoid overflow
    img = np.array(Image.open(path).convert('1'))
    return (img > 0).astype(np.int32)  # Convert to int32 to prevent overflow

def save_binary_image(array, path):
    # Ensure values are in proper range for saving
    binary_array = (array > 0).astype(np.uint8)
    img = Image.fromarray(binary_array * 255)
    img.save(path)

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd_hdexe(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

def calculate_bit_entropy(bits):
    if len(bits) == 0:
        return 0
    ones = sum(bits)
    zeros = len(bits) - ones
    if ones == 0 or zeros == 0:
        return 0
    p1 = ones / len(bits)
    p0 = zeros / len(bits)
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

def calculate_bit_stability(bits):
    """Calculate how stable a bit position is across samples"""
    if len(bits) < 2:
        return 0
    mean_val = np.mean(bits)
    # Stability is higher when bits are consistently 0 or 1
    return abs(mean_val - 0.5) * 2  # Normalize to [0,1]

def calculate_discriminative_power(bit_vals, subject_indices):
    """Calculate how well this bit can discriminate between subjects"""
    if len(set(subject_indices)) < 2:
        return 0
    
    subject_means = {}
    for subj in set(subject_indices):
        subj_bits = [bit_vals[i] for i, s in enumerate(subject_indices) if s == subj]
        if len(subj_bits) > 0:
            subject_means[subj] = np.mean(subj_bits)
    
    if len(subject_means) < 2:
        return 0
    
    # Calculate variance between subject means
    means = list(subject_means.values())
    return np.var(means)

def evaluate_matches_with_details(code_path, mask_path, templates, verbose=True):
    matched_templates = []
    matched_subjects = set()
    hd_values = []
    subject_matches = defaultdict(int)
    subject_totals = defaultdict(int)

    for c, m, base, subject in templates:
        hd = calculate_hd_hdexe(code_path, c, mask_path, m)
        if hd is not None and hd > 0:
            hd_values.append(hd)
            subject_totals[subject] += 1
            if hd <= threshold:
                matched_templates.append((c, m, base, subject))
                matched_subjects.add(subject)
                subject_matches[subject] += 1

    total_templates = len(templates)
    total_subjects = len(set(subject for _, _, _, subject in templates))

    if verbose and hd_values:
        print("Match statistics:")
        print(f" - Total comparisons: {len(hd_values)}")
        print(f" - Average HD: {np.mean(hd_values):.4f}")
        print(f" - Min HD: {np.min(hd_values):.4f}")
        print(f" - Max HD: {np.max(hd_values):.4f}")
        print(f" - Matches under threshold ({threshold}): {sum(1 for hd in hd_values if hd <= threshold)}")
        print(f" - Matched subjects: {len(matched_subjects)} / {total_subjects}")

    return len(matched_templates), len(matched_subjects), hd_values

# === IMPROVED ML DATASET BUILDING ===
def build_enhanced_bit_dataset(codes, masks, templates, min_samples=3):
    """Build a more comprehensive dataset for bit quality classification"""
    H, W = codes[0].shape
    data = []
    labels = []
    positions = []

    # Group by subject for intra/inter class analysis
    subject_codes = defaultdict(list)
    for i, (_, _, _, subject) in enumerate(templates):
        subject_codes[subject].append(i)

    print(f"Processing {H}x{W} bit positions...")
    
    for i in tqdm(range(H), desc="Processing rows"):
        for j in range(W):
            # Get valid bits (where mask is 1)
            valid_indices = [idx for idx, (c, m) in enumerate(zip(codes, masks)) if m[i, j] == 1]
            
            if len(valid_indices) < min_samples:
                continue
                
            bit_vals = [codes[idx][i, j] for idx in valid_indices]
            subject_indices = [templates[idx][3] for idx in valid_indices]
            
            # Calculate features
            entropy = calculate_bit_entropy(bit_vals)
            stability = calculate_bit_stability(bit_vals)
            discriminative = calculate_discriminative_power(bit_vals, subject_indices)
            
            # Calculate intra/inter subject differences with proper type handling
            intra_diffs = []
            inter_diffs = []
            
            for s1, indices1 in subject_codes.items():
                valid_s1 = [idx for idx in indices1 if idx in valid_indices]
                if len(valid_s1) < 2:
                    continue
                    
                values1 = [int(codes[idx][i, j]) for idx in valid_s1]  # Explicit int conversion
                # Intra-subject differences
                for a in range(len(values1)):
                    for b in range(a+1, len(values1)):
                        diff = abs(values1[a] - values1[b])
                        intra_diffs.append(diff)

                # Inter-subject differences
                for s2, indices2 in subject_codes.items():
                    if s1 >= s2:  # Avoid duplicate comparisons
                        continue
                    valid_s2 = [idx for idx in indices2 if idx in valid_indices]
                    if len(valid_s2) == 0:
                        continue
                        
                    values2 = [int(codes[idx][i, j]) for idx in valid_s2]  # Explicit int conversion
                    for v1 in values1:
                        for v2 in values2:
                            diff = abs(v1 - v2)
                            inter_diffs.append(diff)

            if len(intra_diffs) == 0 or len(inter_diffs) == 0:
                continue

            # Enhanced feature vector
            feature_vector = [
                entropy,                    # Bit entropy
                stability,                  # Bit stability
                discriminative,             # Discriminative power
                np.mean(intra_diffs),      # Average intra-subject difference
                np.mean(inter_diffs),      # Average inter-subject difference  
                np.var(bit_vals),          # Bit variance
                len(bit_vals),             # Number of valid samples
                np.mean(bit_vals),         # Bit mean (bias)
            ]

            # Label: 1 if good discriminative bit, 0 otherwise
            # Adjusted criteria for better bit selection
            separability = np.mean(inter_diffs) - np.mean(intra_diffs)
            quality_label = 1 if (separability > 0.05 and entropy > 0.3 and 
                                discriminative > 0.1) else 0

            data.append(feature_vector)
            labels.append(quality_label)
            positions.append((i, j))

    return np.array(data), np.array(labels), positions

def train_enhanced_random_forest(data, labels):
    """Train an enhanced Random Forest model"""
    print(f"Training on {len(data)} samples...")
    print(f"Positive class ratio: {np.mean(labels):.3f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Enhanced Random Forest with better parameters
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\nTraining accuracy: {accuracy_score(y_train, clf.predict(X_train)):.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_names = ['entropy', 'stability', 'discriminative', 'intra_diff', 
                    'inter_diff', 'variance', 'sample_count', 'mean_bias']
    importance = clf.feature_importances_
    print("\nFeature importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.3f}")
    
    return clf

def generate_master_from_enhanced_model(model, codes, masks, templates, data, positions):
    H, W = codes[0].shape
    max_bits = 800
    min_conf = 0.3

    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    confidence_scores = [np.max(prob) for prob in prediction_probs]
    position_confidence = list(zip(positions, predictions, confidence_scores, data))

    positive_positions = [(pos, pred, conf, feat) for pos, pred, conf, feat in position_confidence if pred == 1]
    positive_positions.sort(key=lambda x: x[2], reverse=True)

    best_score = -1
    best_master = None
    best_mask = None

    print("Evaluating top bit combinations by HD matching score + balance...")

    # Try combinations with increasing number of bits
    for cutoff in range(200, max_bits + 1, 100):
        ones = 0
        zeros = 0
        master = np.zeros((H, W), dtype=np.int32)
        master_mask = np.zeros((H, W), dtype=np.int32)
        selected = 0

        for pos, pred, conf, feat in positive_positions[:cutoff]:
            i, j = pos
            valid_bits = [int(codes[k][i, j]) for k, (c, m) in enumerate(zip(codes, masks)) if m[i, j] == 1]
            if len(valid_bits) == 0:
                continue
            bit = 1 if sum(valid_bits) >= len(valid_bits) / 2 else 0
            master[i, j] = bit
            master_mask[i, j] = 1
            if bit == 1:
                ones += 1
            else:
                zeros += 1
            selected += 1

        if selected < 100:
            continue

        matches, total = evaluate_template_match_score(master, master_mask, codes, masks, templates, threshold=0.32)
        one_ratio = ones / (ones + zeros + 1e-6)
        balance_penalty = abs(one_ratio - 0.5)
        score = matches - balance_penalty * selected * 0.5  # Weighted fitness

        print(f"Bits: {selected}, Matches: {matches}, 1-ratio: {one_ratio:.2f}, Score: {score:.2f}")

        if score > best_score:
            best_score = score
            best_master = master.copy()
            best_mask = master_mask.copy()

    if best_master is None:
        print("❌ No valid master generated.")
        return np.zeros((H, W), dtype=np.int32), np.zeros((H, W), dtype=np.int32)

    print(f"✅ Best configuration: Score = {best_score:.2f}")
    return best_master, best_mask

def evaluate_template_match_score(master, master_mask, codes, masks, templates, threshold):
    def save_temp(array):
        img = Image.fromarray((array > 0).astype(np.uint8) * 255)
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(temp.name)
        return temp.name

    code_path = save_temp(master)
    mask_path = save_temp(master_mask)

    matches = 0
    total = 0
    for c, m, _, _ in templates:
        hd = calculate_hd_hdexe(code_path, c, mask_path, m)
        if hd is not None and hd > 0:
            total += 1
            if hd <= threshold:
                matches += 1

    os.remove(code_path)
    os.remove(mask_path)
    return matches, total

# === DATA LOADING ===
def load_all_templates(dataset_root, eye):
    templates = []
    subject_templates = defaultdict(list)
    
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        # CASIA format (L/R subdirectories)
        if 'L' in subdirs or 'R' in subdirs:
            eye_path = os.path.join(subject_path, eye)
            if not os.path.isdir(eye_path):
                continue
            for file in os.listdir(eye_path):
                if file.endswith("_code.png"):
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(eye_path, base + "_code.png")
                    mask_path = os.path.join(eye_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base, subject))
                        subject_templates[subject].append((code_path, mask_path, base))
        
        # IITD format (direct files with eye indicator)
        else:
            for file in os.listdir(subject_path):
                if file.endswith("_code.png") and f"_{eye}" in file:
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(subject_path, base + "_code.png")
                    mask_path = os.path.join(subject_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base, subject))
                        subject_templates[subject].append((code_path, mask_path, base))
    
    return templates, subject_templates

# === MAIN ENTRYPOINT ===
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['L', 'R']:
        print("Please specify the eye: python generate-mastercode-ml.py L or R")
        return

    eye = sys.argv[1]
    print(f"Generating enhanced ML-based master for eye: {eye}")

    output_dir = os.path.join(output_root, f"master_{eye}")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading templates...")
    templates, subject_templates = load_all_templates(dataset_root, eye)
    print(f"Loaded {len(templates)} templates from {len(subject_templates)} subjects")

    if len(templates) == 0:
        print("No templates found!")
        return

    print("Loading binary images...")
    codes = []
    masks = []
    for i, (c, m, b, s) in enumerate(tqdm(templates, desc="Loading images")):
        codes.append(load_binary_image(c))
        masks.append(load_binary_image(m))

    print("Building enhanced dataset for ML model...")
    data, labels, positions = build_enhanced_bit_dataset(codes, masks, templates)
    print(f"Dataset shape: {data.shape}")
    print(f"Positive samples: {np.sum(labels)} ({np.mean(labels):.3f})")

    if len(data) == 0:
        print("No valid data points generated!")
        return

    print("Training enhanced Random Forest model...")
    model = train_enhanced_random_forest(data, labels)

    print("Generating master template using model predictions...")
    master, master_mask = generate_master_from_enhanced_model(
        model, codes, masks, templates, data, positions
    )

    # Save results
    output_path = os.path.join(output_dir, "enhanced_ml_master_code.png")
    output_mask_path = os.path.join(output_dir, "enhanced_ml_master_codemask.png")
    save_binary_image(master, output_path)
    save_binary_image(master_mask, output_mask_path)

    print(f"\nSaved master template to: {output_path}")
    print(f"Saved master mask to: {output_mask_path}")

    print("\nEvaluating final master template...")
    evaluate_matches_with_details(output_path, output_mask_path, templates)

if __name__ == "__main__":
    main()