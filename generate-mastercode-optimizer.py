import os
import re
import subprocess
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode")
seed_code_path = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode/S5925L04_code.png")  # initial IrisCode seed
threshold = 0.32  # Hamming distance threshold

# === UTILITIES ===

def load_binary_image(path):
    return np.array(Image.open(path).convert('1')).astype(np.uint8)

def save_binary_image(array, path):
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)

def run_command(cmd):
    normed = [os.path.normpath(str(c)) for c in cmd]
    result = subprocess.run(normed, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return result.stdout

def calculate_hd_hdexe(code1, code2, mask1, mask2):
    cmd = [os.path.join(tool_path, "hd.exe"), "-i", code1, code2, "-m", mask1, mask2]
    out = run_command(cmd)
    m = re.search(r"=\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None

# === DATA LOADING ===
from collections import defaultdict

def load_all_templates(root, eye='L'):
    templates, subject_templates = [], defaultdict(list)
    for subject in sorted(os.listdir(root)):
        sp = os.path.join(root, subject)
        if not os.path.isdir(sp): continue
        # CASIA style
        for lvl in ['L','R']:
            eye_path = os.path.join(sp, lvl)
            if os.path.isdir(eye_path):
                if lvl != eye: continue
                for f in os.listdir(eye_path):
                    if f.endswith('_code.png'):
                        base = f.replace('_code.png','')
                        cp, mp = os.path.join(eye_path, base+'_code.png'), os.path.join(eye_path, base+'_codemask.png')
                        if os.path.exists(cp) and os.path.exists(mp):
                            templates.append((cp, mp, subject))
                            subject_templates[subject].append((cp, mp))
                break
        else:
            # IITD style
            for f in os.listdir(sp):
                if f.endswith('_code.png') and f.endswith(f"_{eye}_code.png"):
                    base = f.replace('_code.png','')
                    cp, mp = os.path.join(sp, base+'_code.png'), os.path.join(sp, base+'_codemask.png')
                    if os.path.exists(cp) and os.path.exists(mp):
                        templates.append((cp, mp, subject))
                        subject_templates[subject].append((cp, mp))
    return templates, subject_templates

# === OPTIMIZER ===
class GeneticMasterIrisCodeOptimizer:
    def __init__(self, templates, threshold, pop_size=30, gens=50, cx_rate=0.8, mut_rate=0.02):
        self.templates = templates
        self.threshold = threshold
        self.pop_size = pop_size
        self.gens = gens
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.code_length = 1280
        # compute union mask of all masks
        mask_arrays = [load_binary_image(mp) for _, mp, _ in templates]
        combined = np.logical_or.reduce(mask_arrays).astype(np.uint8)
        self.global_mask = combined
        os.makedirs(output_root, exist_ok=True)
        save_binary_image(self.global_mask, os.path.join(output_root, 'global_mask.png'))
        self.global_mask_path = os.path.join(output_root, 'global_mask.png')
        # load seed IrisCode
        seed_arr = load_binary_image(seed_code_path).astype(np.uint8).flatten()
        self.seed = seed_arr

    def initialize_population(self):
        pop = []
        # include seed as first individual
        pop.append(self.seed.copy())
        # fill rest randomly
        for _ in range(self.pop_size - 1):
            indiv = (np.random.rand(self.code_length) < 0.5).astype(np.uint8)
            pop.append(indiv)
        return pop

    def fitness(self, indiv):
        # enforce bit-ratio constraint
        p1 = indiv.mean()
        if p1 < 0.2 or p1 > 0.8:
            return 0.0
        # save candidate code
        save_binary_image(indiv.reshape((1, self.code_length)).astype(np.uint8), 
                          os.path.join(output_root, 'candidate_code.png'))
        matches = 0
        total = len(self.templates)
        for cp, mp, _ in self.templates:
            hd = calculate_hd_hdexe(os.path.join(output_root, 'candidate_code.png'), cp,
                                     self.global_mask_path, mp)
            # count only if hd > 0 and within threshold
            if hd is not None and 0.0 < hd <= self.threshold:
                matches += 1
        return matches / total

    def select_parents(self, pop, fitnesses):
        parents = []
        for _ in range(self.pop_size):
            i, j = random.sample(range(self.pop_size), 2)
            parents.append(pop[i] if fitnesses[i] > fitnesses[j] else pop[j])
        return parents

    def crossover(self, p1, p2):
        if random.random() > self.cx_rate:
            return p1.copy(), p2.copy()
        pt = random.randint(1, self.code_length - 1)
        return (np.concatenate([p1[:pt], p2[pt:]]).astype(np.uint8),
                np.concatenate([p2[:pt], p1[pt:]]).astype(np.uint8))

    def mutate(self, indiv):
        mask = np.random.rand(self.code_length) < self.mut_rate
        indiv[mask] = 1 - indiv[mask]
        return indiv

    def run(self):
        pop = self.initialize_population()
        history = []
        best, best_fit = None, 0.0
        for gen in range(self.gens):
            fits = [self.fitness(ind) for ind in pop]
            gen_best = max(fits)
            idx = fits.index(gen_best)
            if gen_best > best_fit:
                best_fit = gen_best
                best = pop[idx].copy()
            history.append(best_fit)
            print(f"Gen {gen+1}: best match-rate = {best_fit:.3f}")
            parents = self.select_parents(pop, fits)
            next_pop = []
            for i in range(0, self.pop_size, 2):
                c1, c2 = self.crossover(parents[i], parents[i+1])
                next_pop.extend([self.mutate(c1), self.mutate(c2)])
            pop = next_pop
        # save results
        save_binary_image(best.reshape((1, self.code_length)),
                          os.path.join(output_root, 'master_iris_code.png'))
        plt.figure()
        plt.plot(range(1, self.gens+1), history)
        plt.xlabel('Generation')
        plt.ylabel('Best Match Rate')
        plt.title('Optimization Progress')
        plt.tight_layout()
        plt.savefig(os.path.join(output_root, 'optimization_progress.png'))
        print(f"Optimization complete. Best match-rate: {best_fit:.3f}")
        print(f"Master code: master_iris_code.png; Global mask: global_mask.png; History plot: optimization_progress.png")

# === MAIN ===
if __name__ == '__main__':
    templates, _ = load_all_templates(dataset_root, eye='L')
    optimizer = GeneticMasterIrisCodeOptimizer(templates, threshold,
                                               pop_size=30, gens=2,
                                               cx_rate=0.8, mut_rate=0.02)
    optimizer.run()
