import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random
import subprocess
import re
from collections import defaultdict

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-hybrid-iriscode")
threshold = 0.32

# === UTILITY FUNCTIONS ===
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

def calculate_bit_entropy(bits):
    ones = sum(bits)
    zeros = len(bits) - ones
    if ones == 0 or zeros == 0:
        return 0
    p1 = ones / len(bits)
    p0 = zeros / len(bits)
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

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

def evaluate_matches_with_details_from_arrays(master, master_mask, codes, masks, templates, verbose=True):
    from tempfile import NamedTemporaryFile

    def save_temp_img(array):
        img = Image.fromarray((array * 255).astype(np.uint8))
        f = NamedTemporaryFile(delete=False, suffix=".png")
        img.save(f.name)
        return f.name

    master_path = save_temp_img(master)
    mask_path = save_temp_img(master_mask)

    result = evaluate_matches_with_details(master_path, mask_path, templates, verbose=verbose)

    os.remove(master_path)
    os.remove(mask_path)
    return result

# === HYBRID MASTER GENERATOR ===
def generate_hybrid_master(codes, masks, templates, max_generations=5, population_size=20, elite_count=3):
    H, W = codes[0].shape
    N = H * W

    entropy_mask = np.ones((H, W), dtype=np.uint8)
    entropy_values = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            bits = [codes[k][i, j] for k in range(len(codes)) if masks[k][i, j] == 1]
            entropy_mask[i, j] = 1  # Use all bits for now


    valid_indices = np.argwhere(entropy_mask == 1)
    num_valid = len(valid_indices)
    print(f"{num_valid} stable bits selected for optimization.")

    population = [np.random.randint(0, 2, num_valid, dtype=np.uint8) for _ in range(population_size)]

    def decode_flattened(flat_bits):
        master = np.zeros((H, W), dtype=np.uint8)
        for idx, (i, j) in enumerate(valid_indices):
            master[i, j] = flat_bits[idx]
        return master

    def evaluate_individual(flat_bits):
        master = decode_flattened(flat_bits)
        master_mask = entropy_mask.copy()
        matches, subjects, _ = evaluate_matches_with_details_from_arrays(master, master_mask, codes, masks, templates, verbose=False)

        total_bits = np.sum(master_mask)
        ones = np.sum(master * master_mask)
        one_ratio = ones / total_bits
        zero_ratio = 1 - one_ratio

        penalty_weight = 50  # Try tuning this
        fitness = matches - (abs(one_ratio - 0.5) * penalty_weight)

        print(f"   -> Bit distribution: 1s = {ones}, 0s = {total_bits - ones} | 1 ratio = {one_ratio:.2f}, 0 ratio = {zero_ratio:.2f}")

        return fitness, matches, abs(one_ratio - 0.5)



    best_fitness = -np.inf
    best_individual = None

    for gen in tqdm(range(max_generations), desc="Genetic algorithm progress"):
        scored = [(evaluate_individual(ind), ind) for ind in population]
        scored.sort(reverse=True, key=lambda x: x[0][0])

        best_gen_fit, best_gen_matches, best_gen_imb = scored[0][0]
        print(f"Generation {gen+1}: Matches = {best_gen_matches}, Balance = {1 - abs(0.5 - best_gen_imb):.2f}, Fitness = {best_gen_fit:.2f}")

        if best_gen_fit > best_fitness:
            best_fitness = best_gen_fit
            best_individual = scored[0][1]

        elites = [ind for (_, ind) in scored[:elite_count]]
        children = []
        while len(children) < population_size - elite_count:
            p1, p2 = random.sample(elites, 2)
            cp = random.randint(0, num_valid - 1)
            child = np.concatenate((p1[:cp], p2[cp:]))
            for _ in range(num_valid // 20):
                flip_idx = random.randint(0, num_valid - 1)
                child[flip_idx] ^= 1
            children.append(child)

        population = elites + children

    final_master = decode_flattened(best_individual)
    final_mask = entropy_mask.copy()
    return final_master, final_mask

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
        print("Please specify the eye: python generate-mastercode-hybrid.py L or R")
        return

    eye = sys.argv[1]
    print(f"Generating hybrid master for eye: {eye}")

    output_dir = os.path.join(output_root, f"master_casia_{eye}")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading templates...")
    templates, subject_templates = load_all_templates(dataset_root, eye)
    print(f"Loaded {len(templates)} templates from {len(subject_templates)} subjects")

    if len(templates) == 0:
        print("No templates found!")
        return

    codes = [load_binary_image(c) for c, m, b, s in templates]
    masks = [load_binary_image(m) for c, m, b, s in templates]

    master, master_mask = generate_hybrid_master(codes, masks, templates)

    output_path = os.path.join(output_dir, "hybrid_master_code.png")
    output_mask_path = os.path.join(output_dir, "hybrid_master_codemask.png")
    save_binary_image(master, output_path)
    save_binary_image(master_mask, output_mask_path)

    print("Evaluating final master template...")
    evaluate_matches_with_details(output_path, output_mask_path, templates)

if __name__ == "__main__":
    main()