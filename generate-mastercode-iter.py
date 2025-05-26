import os
import re
import subprocess
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode")
threshold = 0.32

# === UTILITIES ===

def load_binary_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('1')
    return np.array(img).astype(np.uint8)

def save_binary_image(array: np.ndarray, path: str):
    """
    Save a binary code array as a 1×N or H×W image, matching sample iriscode format.
    If input is 1D, reshape to (1, length) for a single-row image.
    """
    # Ensure 2D array: single row
    if array.ndim == 1:
        arr2 = array[np.newaxis, :]
    else:
        arr2 = array
    # Convert to 8-bit (0/255)
    img = Image.fromarray((arr2 * 255).astype(np.uint8))
    img.save(path)

def run_command(cmd):
    normed_cmd = [os.path.normpath(str(c)) for c in cmd]
    return subprocess.run(normed_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def calculate_hd_hdexe(code1: str, code2: str, mask1: str, mask2: str) -> float:
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    res = run_command(cmd)
    m = re.search(r"=\s*([0-9.]+)", res.stdout)
    return float(m.group(1)) if m else None

# === DATA LOADING ===

def load_all_templates(root: str, eye: str):
    templates = []  # (code_path, mask_path)
    for subj in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path): continue
        # CASIA format
        eye_dir = os.path.join(subj_path, eye)
        if os.path.isdir(eye_dir):
            for f in os.listdir(eye_dir):
                if f.endswith("_code.png"):
                    base = f.replace("_code.png", "")
                    c = os.path.join(eye_dir, base + "_code.png")
                    m = os.path.join(eye_dir, base + "_codemask.png")
                    if os.path.exists(c) and os.path.exists(m):
                        templates.append((c, m))
        else:
            # IITD-style
            for f in os.listdir(subj_path):
                if f.endswith("_code.png") and f"_{eye}" in f:
                    base = f.replace("_code.png", "")
                    c = os.path.join(subj_path, base + "_code.png")
                    m = os.path.join(subj_path, base + "_codemask.png")
                    if os.path.exists(c) and os.path.exists(m):
                        templates.append((c, m))
    return templates

# === FITNESS & EVALUATION ===

def compute_similarity_binary(master: np.ndarray, templates: np.ndarray) -> float:
    # average similarity = 1 - mean Hamming distance
    hd = np.mean(master != templates, axis=1)
    return float(np.mean(1.0 - hd))

def evaluate_metrics(
    master: np.ndarray,
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresh: float
) -> Tuple[float, float, float]:
    sim_g = 1.0 - np.mean(master != genuine, axis=1)
    sim_i = 1.0 - np.mean(master != impostor, axis=1)
    ta = np.sum(sim_g >= thresh)
    fr = np.sum(sim_g < thresh)
    fa = np.sum(sim_i >= thresh)
    tr = np.sum(sim_i < thresh)
    acc = (ta + tr) / (len(genuine) + len(impostor))
    far = fa / len(impostor)
    frr = fr / len(genuine)
    return acc, far, frr

# === WOLF SELECTION ===

def wolf_selection(
    templates: np.ndarray,
    n_wolves: int = 20
) -> Tuple[np.ndarray, List[float]]:
    """
    Generate random wolves, evaluate average similarity, pick best.
    Returns best wolf and fitness list.
    """
    n_bits = templates.shape[1]
    bit_freq = templates.mean(axis=0)
    wolves = (np.random.rand(n_wolves, n_bits) < bit_freq).astype(int)
    fitness = [compute_similarity_binary(w, templates) for w in wolves]
    best_idx = int(np.argmax(fitness))
    return wolves[best_idx], fitness

# === OPTIMIZERS ===

def optimize_genetic(
    templates: np.ndarray,
    seed: np.ndarray = None,
    pop_size: int = 50,
    n_gen: int = 100,
    cx_prob: float = 0.7,
    mut_rate: float = 0.01
) -> Tuple[np.ndarray, List[float]]:
    n_bits = templates.shape[1]
    bit_freq = templates.mean(axis=0)
    # initialize population
    pop = (np.random.rand(pop_size, n_bits) < bit_freq).astype(int)
    if seed is not None:
        pop[0] = seed.copy()

    history = []
    for _ in range(n_gen):
        fit = np.array([compute_similarity_binary(ind, templates) for ind in pop])
        history.append(fit.max())
        # tournament
        new = []
        for __ in range(pop_size):
            i, j = np.random.randint(pop_size, size=2)
            new.append(pop[i] if fit[i] > fit[j] else pop[j])
        new = np.array(new)
        # crossover
        for i in range(0, pop_size, 2):
            if np.random.rand() < cx_prob and i+1 < pop_size:
                pt = np.random.randint(1, n_bits)
                a, b = new[i].copy(), new[i+1].copy()
                new[i, pt:], new[i+1, pt:] = b[pt:], a[pt:]
        # mutate
        m = (np.random.rand(pop_size, n_bits) < mut_rate)
        pop = np.bitwise_xor(new, m.astype(int))
    best = pop[int(np.argmax([compute_similarity_binary(x, templates) for x in pop]))]
    return best, history


def optimize_annealing(
    templates: np.ndarray,
    seed: np.ndarray = None,
    n_iter: int = 1000,
    temp0: float = 1.0,
    cool: float = 0.995
) -> Tuple[np.ndarray, List[float]]:
    n_bits = templates.shape[1]
    if seed is None:
        bit_freq = templates.mean(axis=0)
        current = (np.random.rand(n_bits) < bit_freq).astype(int)
    else:
        current = seed.copy()
    curr_score = compute_similarity_binary(current, templates)
    hist = [curr_score]
    T = temp0
    for _ in range(n_iter):
        nxt = current.copy()
        i = np.random.randint(n_bits)
        nxt[i] ^= 1
        s = compute_similarity_binary(nxt, templates)
        d = s - curr_score
        if d > 0 or np.random.rand() < np.exp(d/T):
            current, curr_score = nxt, s
        hist.append(curr_score)
        T *= cool
    return current, hist

# === MAIN EXECUTION ===

if __name__ == '__main__':
    os.makedirs(output_root, exist_ok=True)
    for eye in ['L','R']:
        tpl = load_all_templates(dataset_root, eye)
        # Load binary arrays (flatten 1280x1 to 1280)
        codes = []
        for cpath, mpath in tpl:
            img = load_binary_image(cpath)
            # flatten row vector to 1D
            codes.append(img.flatten())
        arr = np.stack(codes, axis=0)
        # Wolf selection
        seed, wolf_hist = wolf_selection(arr, n_wolves=30)
        # Genetic
        ga_master, ga_hist = optimize_genetic(arr, seed=seed, pop_size=100, n_gen=200)
        # Annealing
        sa_master, sa_hist = optimize_annealing(arr, seed=seed, n_iter=2000)
        # Save masters
        save_binary_image(ga_master, os.path.join(output_root, f'master_GA_{eye}.png'))
        save_binary_image(sa_master, os.path.join(output_root, f'master_SA_{eye}.png'))
        # Plot
        plt.figure(); plt.plot(wolf_hist, label='Wolf init'); plt.plot(ga_hist, label='GA'); plt.plot(sa_hist, label='SA');
        plt.title(f'{eye}-eye optimization'); plt.legend(); plt.savefig(os.path.join(output_root, f'opt_{eye}.png'))
        # Evaluate
        # (use half templates as impostor for demo)
        half = len(arr)//2
        acc_g, far_g, frr_g = evaluate_metrics(ga_master, arr[:half], arr[half:], threshold)
        acc_s, far_s, frr_s = evaluate_metrics(sa_master, arr[:half], arr[half:], threshold)
        print(f"{eye}: GA Acc={acc_g:.3f},FAR={far_g:.3f},FRR={frr_g:.3f}")
        print(f"{eye}: SA Acc={acc_s:.3f},FAR={far_s:.3f},FRR={frr_s:.3f}")
