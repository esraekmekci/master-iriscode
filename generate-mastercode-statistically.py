import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random
import subprocess
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode")
threshold = 0.32

# === UTILITIES ===

def load_binary_image(path):
    return np.array(Image.open(path).convert('1')).astype(np.uint8)

def save_binary_image(array, path):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd_hdexe(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    result = run_command(cmd)
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None

# === DATA LOADING AND ANALYSIS ===

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

def analyze_dataset_distances(templates, sample_size=50):
    """
    Analyze the distribution of Hamming distances in the dataset
    """
    print("\n🔍 Analyzing dataset Hamming distance distribution...")
    
    # Sample random pairs for analysis
    sampled_templates = random.sample(templates, min(sample_size, len(templates)))
    distances = []
    intra_subject_distances = []
    inter_subject_distances = []
    
    temp_files = []
    
    for i, (c1, m1, _, s1) in enumerate(sampled_templates):
        for j, (c2, m2, _, s2) in enumerate(sampled_templates[i+1:], i+1):
            hd = calculate_hd_hdexe(c1, c2, m1, m2)
            if hd is not None and hd > 0:
                distances.append(hd)
                if s1 == s2:
                    intra_subject_distances.append(hd)
                else:
                    inter_subject_distances.append(hd)
    
    if distances:
        print(f"📊 Dataset Analysis Results:")
        print(f"   - Total distance samples: {len(distances)}")
        print(f"   - Average HD: {np.mean(distances):.4f}")
        print(f"   - Min HD: {np.min(distances):.4f}")
        print(f"   - Max HD: {np.max(distances):.4f}")
        print(f"   - Std HD: {np.std(distances):.4f}")
        
        if intra_subject_distances:
            print(f"   - Intra-subject avg: {np.mean(intra_subject_distances):.4f}")
        if inter_subject_distances:
            print(f"   - Inter-subject avg: {np.mean(inter_subject_distances):.4f}")
        
        # Suggest realistic threshold
        min_hd = np.min(distances)
        suggested_threshold = min_hd + (np.mean(distances) - min_hd) * 0.5
        print(f"\n💡 Suggested threshold: {suggested_threshold:.4f} (current: {threshold})")
        
        return suggested_threshold, distances
    
    return None, []

# === ADVANCED MASTER GENERATION STRATEGIES ===

def generate_entropy_based_master(codes, masks, templates):
    """
    Generate master using entropy-based approach focusing on most stable bits
    """
    H, W = codes[0].shape
    master = np.zeros((H, W), dtype=np.uint8)
    master_mask = np.ones((H, W), dtype=np.uint8)
    
    # Group by subjects
    subject_codes = defaultdict(list)
    for i, (c, m, base, subject) in enumerate(templates):
        subject_codes[subject].append(i)
    
    print(f"🧮 Generating entropy-based master template...")
    
    for i in tqdm(range(H), desc="Computing bit entropy"):
        for j in range(W):
            bit_values = []
            subject_bits = []
            
            # Collect one representative bit per subject
            for subject, indices in subject_codes.items():
                subject_bit_values = []
                for idx in indices:
                    if masks[idx][i, j] == 1:
                        subject_bit_values.append(codes[idx][i, j])
                
                if subject_bit_values:
                    # Take majority vote within subject
                    subject_bit = 1 if sum(subject_bit_values) >= len(subject_bit_values) / 2 else 0
                    subject_bits.append(subject_bit)
                    bit_values.extend(subject_bit_values)
            
            if not subject_bits:
                master_mask[i, j] = 0
                continue
            
            # Calculate entropy/stability of this bit position
            ones_count = sum(subject_bits)
            zeros_count = len(subject_bits) - ones_count
            
            # If this bit position has very high variance, mask it out
            if len(subject_bits) > 2:
                entropy = calculate_bit_entropy(subject_bits)
                if entropy > 0.9:  # High entropy = unstable bit
                    master_mask[i, j] = 0
                    continue
            
            # Choose the most common value
            if ones_count > zeros_count:
                master[i, j] = 1
            elif zeros_count > ones_count:
                master[i, j] = 0
            else:
                # For ties, use template-level majority
                if bit_values:
                    master[i, j] = 1 if sum(bit_values) >= len(bit_values) / 2 else 0
                else:
                    master[i, j] = 0
    
    return master, master_mask

def calculate_bit_entropy(bits):
    """Calculate Shannon entropy of a bit sequence"""
    if len(bits) == 0:
        return 0
    
    ones = sum(bits)
    zeros = len(bits) - ones
    
    if ones == 0 or zeros == 0:
        return 0
    
    p1 = ones / len(bits)
    p0 = zeros / len(bits)
    
    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    return entropy

def generate_cluster_based_master(codes, masks, templates, n_clusters=3):
    """
    Generate multiple master templates based on clustering
    """
    print(f"🎯 Generating cluster-based masters ({n_clusters} clusters)...")
    
    # Simple clustering based on bit similarity
    subject_codes = defaultdict(list)
    for i, (c, m, base, subject) in enumerate(templates):
        subject_codes[subject].append(i)
    
    # Calculate subject representatives
    subject_representatives = {}
    for subject, indices in subject_codes.items():
        if indices:
            # Use first template as representative
            subject_representatives[subject] = indices[0]
    
    # Simple k-means like clustering
    clusters = defaultdict(list)
    subjects = list(subject_representatives.keys())
    
    # Random initial assignment
    for i, subject in enumerate(subjects):
        cluster_id = i % n_clusters
        clusters[cluster_id].append(subject)
    
    masters = []
    for cluster_id in range(n_clusters):
        if not clusters[cluster_id]:
            continue
            
        cluster_indices = []
        for subject in clusters[cluster_id]:
            cluster_indices.extend(subject_codes[subject])
        
        if cluster_indices:
            cluster_codes = [codes[i] for i in cluster_indices]
            cluster_masks = [masks[i] for i in cluster_indices]
            cluster_templates = [(templates[i][0], templates[i][1], templates[i][2], templates[i][3]) for i in cluster_indices]
            
            master, master_mask = generate_entropy_based_master(cluster_codes, cluster_masks, cluster_templates)
            masters.append((master, master_mask, len(cluster_indices)))
    
    return masters

# === EVALUATION FUNCTIONS ===

def evaluate_matches_with_details(code_path, mask_path, templates, verbose=True):
    """
    Detailed evaluation with comprehensive statistics
    """
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
        print(f"📊 Detailed HD Statistics:")
        print(f"   - Count: {len(hd_values)}")
        print(f"   - Average: {np.mean(hd_values):.4f}")
        print(f"   - Minimum: {np.min(hd_values):.4f}")
        print(f"   - 25th percentile: {np.percentile(hd_values, 25):.4f}")
        print(f"   - Median: {np.median(hd_values):.4f}")
        print(f"   - 75th percentile: {np.percentile(hd_values, 75):.4f}")
        print(f"   - Below threshold ({threshold}): {sum(1 for hd in hd_values if hd <= threshold)}")
        
        print(f"\n✅ Results:")
        print(f"   - Matched {len(matched_templates)}/{total_templates} templates ({len(matched_templates)/total_templates*100:.1f}%)")
        print(f"   - Matched {len(matched_subjects)}/{total_subjects} subjects ({len(matched_subjects)/total_subjects*100:.1f}%)")
        
        if matched_subjects:
            print(f"   - Subjects with matches: {sorted(matched_subjects)}")
    
    return len(matched_templates), len(matched_subjects), hd_values

# === MAIN ENTRYPOINT ===

def main():
    global threshold
    if len(sys.argv) != 2 or sys.argv[1] not in ['L', 'R']:
        print("❌ Please specify the eye: python generate-mastercode-statistically.py L or R")
        return

    eye = sys.argv[1]
    print(f"👁️ Generating master template for eye: {eye}")

    output_dir = os.path.join(output_root, f"master_{eye}")
    os.makedirs(output_dir, exist_ok=True)

    print("📂 Loading templates...")
    templates, subject_templates = load_all_templates(dataset_root, eye)
    print(f"✅ Loaded {len(templates)} templates from {len(subject_templates)} subjects for eye '{eye}'")

    if len(templates) == 0:
        print("❌ No templates found!")
        return

    # Analyze dataset first
    suggested_threshold, distances = analyze_dataset_distances(templates)
    
    if suggested_threshold and suggested_threshold > threshold:
        print(f"\n⚠️  WARNING: Your threshold ({threshold}) might be too strict!")
        print(f"💡 Consider using threshold: {suggested_threshold:.4f}")
        
        # Ask user if they want to use suggested threshold
        response = input(f"\nUse suggested threshold {suggested_threshold:.4f}? (y/n): ").lower().strip()
        if response == 'y':
            threshold = suggested_threshold
            print(f"✅ Updated threshold to {threshold:.4f}")

    print("🧠 Loading binary iriscode arrays...")
    codes = [load_binary_image(c) for c, m, b, s in templates]
    masks = [load_binary_image(m) for c, m, b, s in templates]

    # Strategy 1: Single entropy-based master
    print("\n🎯 Strategy 1: Single entropy-based master")
    master, master_mask = generate_entropy_based_master(codes, masks, templates)
    
    output_path = os.path.join(output_dir, "entropy_master_code.png")
    output_mask_path = os.path.join(output_dir, "entropy_master_codemask.png")
    save_binary_image(master, output_path)
    save_binary_image(master_mask, output_mask_path)
    
    print(f"💾 Entropy master saved at: {output_path}")
    matches, subject_matches, hd_vals = evaluate_matches_with_details(output_path, output_mask_path, templates)
    
    best_matches = matches
    best_path = output_path
    best_mask_path = output_mask_path
    
    # Strategy 2: Cluster-based masters
    print("\n🎯 Strategy 2: Cluster-based masters")
    cluster_masters = generate_cluster_based_master(codes, masks, templates, n_clusters=3)
    
    for i, (cluster_master, cluster_mask, cluster_size) in enumerate(cluster_masters):
        cluster_path = os.path.join(output_dir, f"cluster_{i}_master_code.png")
        cluster_mask_path = os.path.join(output_dir, f"cluster_{i}_master_codemask.png")
        
        save_binary_image(cluster_master, cluster_path)
        save_binary_image(cluster_mask, cluster_mask_path)
        
        print(f"\n📊 Cluster {i} (size: {cluster_size}):")
        c_matches, c_subject_matches, c_hd_vals = evaluate_matches_with_details(
            cluster_path, cluster_mask_path, templates, verbose=False
        )
        
        print(f"   - Matches: {c_matches} templates, {c_subject_matches} subjects")
        if c_hd_vals:
            print(f"   - Min HD: {np.min(c_hd_vals):.4f}, Avg HD: {np.mean(c_hd_vals):.4f}")
        
        if c_matches > best_matches:
            best_matches = c_matches
            best_path = cluster_path
            best_mask_path = cluster_mask_path
    
    print(f"\n🏆 Best performing master: {os.path.basename(best_path)}")
    print(f"   - {best_matches} template matches")
    
    # Copy best to standard name
    final_path = os.path.join(output_dir, "master_code.png")
    final_mask_path = os.path.join(output_dir, "master_codemask.png")
    
    if best_path != final_path:
        import shutil
        shutil.copy2(best_path, final_path)
        shutil.copy2(best_mask_path, final_mask_path)
    
    print(f"\n🎯 Final master template saved at: {final_path}")

if __name__ == "__main__":
    main()