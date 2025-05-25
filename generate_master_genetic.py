import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random
import subprocess
import re

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
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

# === DATA LOADING ===

def load_all_templates(dataset_root, eye):
    templates = []
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        if 'L' in subdirs or 'R' in subdirs:
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
            for file in os.listdir(subject_path):
                if file.endswith("_code.png") and f"_{eye}_" in file:
                    base = file.replace("_code.png", "")
                    code_path = os.path.join(subject_path, base + "_code.png")
                    mask_path = os.path.join(subject_path, base + "_codemask.png")
                    if os.path.exists(code_path) and os.path.exists(mask_path):
                        templates.append((code_path, mask_path, base))
    return templates

# === INITIAL MASTER TEMPLATE ===

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

# === GENETIC ALGORITHM ===

def uniform_crossover(parent1, parent2):
    assert parent1.shape == parent2.shape
    H, W = parent1.shape
    mask = np.random.randint(0, 2, size=(H, W), dtype=np.uint8)
    child = (parent1 & mask) | (parent2 & ~mask)
    return child

def tournament_selection(scored_population, k=3):
    contenders = random.sample(scored_population, k)
    contenders.sort(key=lambda x: x[1], reverse=True)
    return contenders[0][0]

def mutate(individual, mask, rate):
    mutated = individual.copy()
    H, W = individual.shape
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and np.random.rand() < rate:
                mutated[i, j] = 1 - mutated[i, j]
    return mutated

def evaluate_fitness(code_array, mask_path, templates):
    temp_path = "temp_eval.png"
    save_binary_image(code_array, temp_path)
    total_hd = 0
    valid_count = 0
    matches = 0
    for c, m, _ in templates:
        hd = calculate_hd_hdexe(temp_path, c, mask_path, m)
        if hd is not None and hd > 0:
            total_hd += hd
            valid_count += 1
            if hd <= threshold:
                matches += 1
    if valid_count == 0:
        return 0
    avg_hd = total_hd / valid_count
    fitness = matches + (1.0 - avg_hd)
    return fitness

def genetic_algorithm_master(codes, masks, master_mask, templates, output_mask_path,
                              population_size=30, generations=1, mutation_rate=0.08,
                              elite_fraction=0.15, tournament_k=4):
    H, W = codes[0].shape
    print("\n🧬 Starting Advanced Genetic Algorithm...")

    base, _ = generate_initial_master_mode(codes, masks)
    population = [base.copy()]
    for _ in range(population_size - 1):
        population.append(mutate(base, master_mask, 0.2))

    elite_count = max(1, int(population_size * elite_fraction))

    for gen in range(generations):
        print(f"\n🔁 Generation {gen+1}/{generations}")
        scored = [(ind, evaluate_fitness(ind, output_mask_path, templates)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_score = scored[0][1]
        print(f"🏆 Best fitness: {best_score:.4f}")

        new_population = [scored[i][0] for i in range(elite_count)]

        while len(new_population) < population_size:
            parent1 = tournament_selection(scored, k=tournament_k)
            parent2 = tournament_selection(scored, k=tournament_k)
            child = uniform_crossover(parent1, parent2)
            child = mutate(child, master_mask, mutation_rate)
            new_population.append(child)

        population = new_population

    best_individual = scored[0][0]
    return best_individual

# === FINAL MATCH EVALUATION ===

def evaluate_matches(code_path, mask_path, templates):
    matched_templates = []
    matched_subjects = set()

    for c, m, base in templates:
        hd = calculate_hd_hdexe(code_path, c, mask_path, m)
        if hd is not None and 0 < hd <= threshold:
            matched_templates.append((c, m, base))

            # Extract subject ID from full path
            subject_id = os.path.basename(os.path.dirname(os.path.dirname(c)))  # goes 2 dirs up
            matched_subjects.add(subject_id)

    total_templates = len(templates)
    total_subjects = len({os.path.basename(os.path.dirname(os.path.dirname(c))) for c, m, b in templates})

    print(f"\n✅ Matched {len(matched_templates)}/{total_templates} templates (HD ≤ {threshold})")
    print(f"✅ These belong to {len(matched_subjects)}/{total_subjects} unique subjects")

    return len(matched_templates)

# === MAIN ENTRYPOINT ===

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['L', 'R']:
        print("❌ Please specify the eye: python generate_master_ga_en.py L or R")
        return

    eye = sys.argv[1]
    print(f"👁️ Generating master template for eye: {eye}")

    output_dir = os.path.join(output_root, f"master_{eye}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_code.png")
    output_mask_path = os.path.join(output_dir, "master_codemask.png")

    print("📂 Loading templates...")
    templates = load_all_templates(dataset_root, eye)
    print(f"✅ Loaded {len(templates)} templates for eye '{eye}'.")

    print("🧠 Loading binary iriscode arrays...")
    codes = [load_binary_image(c) for c, m, b in templates]
    masks = [load_binary_image(m) for c, m, b in templates]

    print("📥 Creating initial master template using bitwise mode...")
    master, master_mask = generate_initial_master_mode(codes, masks)
    save_binary_image(master_mask, output_mask_path)

    print("⚙️ Optimizing template using Genetic Algorithm...")
    master = genetic_algorithm_master(codes, masks, master_mask, templates, output_mask_path)

    save_binary_image(master, output_path)
    print(f"\n💾 Master template saved at: {output_path}")

    evaluate_matches(output_path, output_mask_path, templates)

if __name__ == "__main__":
    main()
