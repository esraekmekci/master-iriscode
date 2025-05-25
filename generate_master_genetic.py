import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random
import subprocess
import re
from collections import defaultdict, Counter
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# === CONFIGURATION ===
tool_path = os.path.abspath(r"C:/Users/PC/Downloads/USITv3.0.0/USITv3.0.0/bin")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/IITD_Database")
dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/samples")
# dataset_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/CASIA-Iris-Syn")
output_root = os.path.abspath(r"C:/Users/PC/Desktop/ceng/term4.2/ceng507/generated-master-iriscode-genetic")
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
    subject_templates = defaultdict(list)
    
    for subject in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        # CASIA format
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
        
        # IITD format
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

# === IRIS-SPECIFIC GENETIC ALGORITHM ===

class IrisGeneticAlgorithm:
    def __init__(self, templates, codes, masks, master_mask, 
                 population_size=50, generations=30, 
                 crossover_rate=0.8, mutation_rate=0.05,
                 elite_ratio=0.1, tournament_size=5):
        
        self.templates = templates
        self.codes = codes
        self.masks = masks
        self.master_mask = master_mask
        self.H, self.W = codes[0].shape
        
        # GA Parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        
        # Subject grouping for fitness
        self.subject_groups = defaultdict(list)
        for i, (_, _, _, subject) in enumerate(templates):
            self.subject_groups[subject].append(i)
        self.num_subjects = len(self.subject_groups)
        
        # Statistics
        self.generation_stats = []
        self.temp_counter = 0
        
        print(f"🧬 Iris GA initialized:")
        print(f"   - Population: {population_size}, Generations: {generations}")
        print(f"   - Crossover: {crossover_rate}, Mutation: {mutation_rate}")
        print(f"   - Elite ratio: {elite_ratio}, Tournament: {tournament_size}")
        print(f"   - Subjects: {self.num_subjects}, Templates: {len(templates)}")

    def initialize_population(self):
        """Initialize population with diverse strategies"""
        population = []
        
        # Strategy 1: Majority voting baseline (10%)
        baseline_count = max(1, self.population_size // 10)
        baseline = self.generate_majority_baseline()
        for _ in range(baseline_count):
            individual = baseline.copy()
            if random.random() < 0.3:  # Add some noise
                individual = self.mutate_individual(individual, rate=0.1)
            population.append(individual)
        
        # Strategy 2: Subject-representative based (30%)
        subject_count = max(1, self.population_size * 3 // 10)
        for _ in range(subject_count):
            individual = self.generate_subject_representative()
            population.append(individual)
        
        # Strategy 3: Random templates as seeds (30%)
        template_count = max(1, self.population_size * 3 // 10)
        for _ in range(template_count):
            individual = self.generate_template_based()
            population.append(individual)
        
        # Strategy 4: Entropy-based (20%)
        entropy_count = max(1, self.population_size // 5)
        for _ in range(entropy_count):
            individual = self.generate_entropy_based()
            population.append(individual)
        
        # Strategy 5: Pure random (10%)
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            individual = self.generate_random()
            population.append(individual)
        
        return population

    def generate_majority_baseline(self):
        """Generate baseline using majority voting"""
        baseline = np.zeros((self.H, self.W), dtype=np.uint8)
        
        for i in range(self.H):
            for j in range(self.W):
                if self.master_mask[i, j] == 1:
                    # Subject-level majority voting
                    subject_votes = []
                    for subject, indices in self.subject_groups.items():
                        subject_bits = [self.codes[idx][i, j] for idx in indices 
                                       if self.masks[idx][i, j] == 1]
                        if subject_bits:
                            subject_vote = 1 if sum(subject_bits) >= len(subject_bits) / 2 else 0
                            subject_votes.append(subject_vote)
                    
                    if subject_votes:
                        baseline[i, j] = 1 if sum(subject_votes) >= len(subject_votes) / 2 else 0
        
        return baseline

    def generate_subject_representative(self):
        """Generate individual based on random subject selection"""
        individual = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Select random subset of subjects
        selected_subjects = random.sample(list(self.subject_groups.keys()), 
                                        max(1, len(self.subject_groups) // 2))
        
        for i in range(self.H):
            for j in range(self.W):
                if self.master_mask[i, j] == 1:
                    votes = []
                    for subject in selected_subjects:
                        indices = self.subject_groups[subject]
                        subject_bits = [self.codes[idx][i, j] for idx in indices 
                                       if self.masks[idx][i, j] == 1]
                        if subject_bits:
                            votes.append(1 if sum(subject_bits) >= len(subject_bits) / 2 else 0)
                    
                    if votes:
                        individual[i, j] = 1 if sum(votes) >= len(votes) / 2 else 0
        
        return individual

    def generate_template_based(self):
        """Generate individual based on random template"""
        # Select random template as base
        base_idx = random.randint(0, len(self.codes) - 1)
        individual = self.codes[base_idx].copy()
        
        # Apply mask
        individual = individual & self.master_mask
        
        # Add some mutations
        individual = self.mutate_individual(individual, rate=0.15)
        
        return individual

    def generate_entropy_based(self):
        """Generate individual focusing on low-entropy positions"""
        individual = np.zeros((self.H, self.W), dtype=np.uint8)
        
        for i in range(self.H):
            for j in range(self.W):
                if self.master_mask[i, j] == 1:
                    # Calculate bit entropy at this position
                    all_bits = [self.codes[idx][i, j] for idx in range(len(self.codes)) 
                               if self.masks[idx][i, j] == 1]
                    
                    if all_bits:
                        ones = sum(all_bits)
                        zeros = len(all_bits) - ones
                        
                        # For high entropy positions, use random
                        if min(ones, zeros) > len(all_bits) * 0.3:
                            individual[i, j] = random.randint(0, 1)
                        else:
                            # For low entropy, use majority
                            individual[i, j] = 1 if ones > zeros else 0
        
        return individual

    def generate_random(self):
        """Generate random individual"""
        individual = np.random.randint(0, 2, (self.H, self.W), dtype=np.uint8)
        return individual & self.master_mask

    def fitness_function(self, individual):
        """
        Advanced fitness function for iris templates
        """
        # Save individual temporarily
        temp_path = f"temp_ga_{self.temp_counter}.png"
        temp_mask_path = f"temp_ga_mask_{self.temp_counter}.png"
        self.temp_counter += 1
        
        save_binary_image(individual, temp_path)
        save_binary_image(self.master_mask, temp_mask_path)
        
        try:
            # Calculate metrics
            match_count = 0
            subject_matches = set()
            distance_sum = 0
            valid_distances = 0
            quality_bonus = 0
            
            # Sample for performance (use subset for large datasets)
            sample_templates = self.templates
            if len(self.templates) > 100:
                sample_templates = random.sample(self.templates, 100)
            
            for c, m, _, subject in sample_templates:
                hd = calculate_hd_hdexe(temp_path, c, temp_mask_path, m)
                if hd is not None and hd > 0:
                    distance_sum += hd
                    valid_distances += 1
                    
                    if hd <= threshold:
                        match_count += 1
                        subject_matches.add(subject)
                    
                    # Quality bonus for near-matches
                    if threshold < hd <= threshold + 0.05:
                        quality_bonus += 0.5
            
            # Multi-objective fitness
            if valid_distances == 0:
                return 0.0
            
            avg_distance = distance_sum / valid_distances
            subject_diversity = len(subject_matches)
            
            # Fitness components
            match_fitness = match_count * 10  # Primary objective
            subject_fitness = subject_diversity * 5  # Subject diversity
            quality_fitness = (1.0 - min(avg_distance, 1.0)) * 2  # Distance quality
            bonus_fitness = quality_bonus  # Near-match bonus
            
            total_fitness = match_fitness + subject_fitness + quality_fitness + bonus_fitness
            
        except Exception as e:
            total_fitness = 0.0
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(temp_mask_path):
                    os.remove(temp_mask_path)
            except:
                pass
        
        return total_fitness

    def tournament_selection(self, population, fitness_scores):
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(self.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx].copy()

    def iris_crossover(self, parent1, parent2):
        """Iris-specific crossover operators"""
        if random.random() < 0.5:
            return self.block_crossover(parent1, parent2)
        else:
            return self.weighted_uniform_crossover(parent1, parent2)

    def block_crossover(self, parent1, parent2):
        """Block-based crossover preserving iris structure"""
        child = np.zeros_like(parent1)
        
        # Divide into blocks
        block_h = max(1, self.H // 4)
        block_w = max(1, self.W // 8)
        
        for i in range(0, self.H, block_h):
            for j in range(0, self.W, block_w):
                # Choose parent for this block
                if random.random() < 0.5:
                    child[i:i+block_h, j:j+block_w] = parent1[i:i+block_h, j:j+block_w]
                else:
                    child[i:i+block_h, j:j+block_w] = parent2[i:i+block_h, j:j+block_w]
        
        return child & self.master_mask

    def weighted_uniform_crossover(self, parent1, parent2):
        """Weighted uniform crossover based on bit stability"""
        child = np.zeros_like(parent1)
        
        for i in range(self.H):
            for j in range(self.W):
                if self.master_mask[i, j] == 1:
                    # Calculate bit stability at this position
                    all_bits = [self.codes[idx][i, j] for idx in range(len(self.codes)) 
                               if self.masks[idx][i, j] == 1]
                    
                    if all_bits:
                        ones_ratio = sum(all_bits) / len(all_bits)
                        stability = 1.0 - 2 * abs(ones_ratio - 0.5)  # 0 = unstable, 1 = stable
                        
                        # For stable positions, prefer consistency
                        if stability > 0.7:
                            if parent1[i, j] == parent2[i, j]:
                                child[i, j] = parent1[i, j]
                            else:
                                child[i, j] = 1 if ones_ratio > 0.5 else 0
                        else:
                            # For unstable positions, random choice
                            child[i, j] = parent1[i, j] if random.random() < 0.5 else parent2[i, j]
        
        return child

    def adaptive_mutation(self, individual, generation):
        """Adaptive mutation rate based on generation"""
        # Decrease mutation rate over generations
        adaptive_rate = self.mutation_rate * (1.0 - generation / self.generations)
        adaptive_rate = max(0.01, adaptive_rate)  # Minimum rate
        
        return self.mutate_individual(individual, adaptive_rate)

    def mutate_individual(self, individual, rate=None):
        """Mutate individual with position-aware mutations"""
        if rate is None:
            rate = self.mutation_rate
        
        mutated = individual.copy()
        
        for i in range(self.H):
            for j in range(self.W):
                if self.master_mask[i, j] == 1 and random.random() < rate:
                    # Bit-level mutation
                    mutated[i, j] = 1 - mutated[i, j]
        
        return mutated

    def run_genetic_algorithm(self):
        """Main GA loop"""
        print(f"\n🚀 Starting Iris Genetic Algorithm...")
        
        # Initialize population
        print("🔄 Initializing population...")
        population = self.initialize_population()
        
        best_individual = None
        best_fitness = -1
        no_improvement_count = 0
        
        for generation in range(self.generations):
            print(f"\n🧬 Generation {generation + 1}/{self.generations}")
            
            # Evaluate fitness
            print("📊 Evaluating fitness...")
            fitness_scores = []
            for i, individual in enumerate(tqdm(population, desc="Fitness evaluation")):
                fitness = self.fitness_function(individual)
                fitness_scores.append(fitness)
            
            # Track best
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
                no_improvement_count = 0
                print(f"✨ New best fitness: {best_fitness:.4f}")
            else:
                no_improvement_count += 1
            
            # Statistics
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness
            })
            
            print(f"📈 Fitness - Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}, Std: {std_fitness:.4f}")
            
            # Early stopping
            if no_improvement_count >= 10:
                print(f"🛑 Early stopping at generation {generation + 1} (no improvement for 10 generations)")
                break
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:self.elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.iris_crossover(parent1, parent2)
                else:
                    child = parent1.copy() if random.random() < 0.5 else parent2.copy()
                
                # Mutation
                child = self.adaptive_mutation(child, generation)
                
                new_population.append(child)
            
            population = new_population
        
        print(f"\n🏆 GA completed! Best fitness: {best_fitness:.4f}")
        return best_individual, best_fitness, self.generation_stats

# === MAIN FUNCTIONS ===

def run_iris_ga(templates, codes, masks, master_mask):
    """Run the iris-specific genetic algorithm"""
    
    ga = IrisGeneticAlgorithm(
        templates=templates,
        codes=codes,
        masks=masks,
        master_mask=master_mask,
        population_size=40,
        generations=25,
        crossover_rate=0.8,
        mutation_rate=0.08,
        elite_ratio=0.15,
        tournament_size=5
    )
    
    best_individual, best_fitness, stats = ga.run_genetic_algorithm()
    
    return best_individual, best_fitness, stats

def evaluate_final_results(master_path, mask_path, templates):
    """Final evaluation of the master template"""
    matched_templates = []
    matched_subjects = set()
    hd_values = []
    
    print(f"\n🎯 Final Evaluation:")
    
    for c, m, base, subject in tqdm(templates, desc="Evaluating"):
        hd = calculate_hd_hdexe(master_path, c, mask_path, m)
        if hd is not None and hd > 0:
            hd_values.append(hd)
            if hd <= threshold:
                matched_templates.append((c, m, base, subject))
                matched_subjects.add(subject)
    
    if hd_values:
        print(f"📊 Results:")
        print(f"   - Total comparisons: {len(hd_values)}")
        print(f"   - Matched templates: {len(matched_templates)}/{len(templates)} ({len(matched_templates)/len(templates)*100:.1f}%)")
        print(f"   - Matched subjects: {len(matched_subjects)}")
        print(f"   - Average HD: {np.mean(hd_values):.4f}")
        print(f"   - Min HD: {np.min(hd_values):.4f}")
        print(f"   - HD std: {np.std(hd_values):.4f}")
        
        if matched_subjects:
            print(f"   - Subjects with matches: {sorted(matched_subjects)}")
    
    return len(matched_templates), len(matched_subjects)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['L', 'R']:
        print("❌ Please specify the eye: python generate_master_genetic.py L or R")
        return

    eye = sys.argv[1]
    print(f"👁️ Generating master template using Iris GA for eye: {eye}")

    output_dir = os.path.join(output_root, f"master_{eye}_ga")
    os.makedirs(output_dir, exist_ok=True)

    print("📂 Loading templates...")
    templates, subject_templates = load_all_templates(dataset_root, eye)
    print(f"✅ Loaded {len(templates)} templates from {len(subject_templates)} subjects")

    if len(templates) == 0:
        print("❌ No templates found!")
        return

    print("🧠 Loading binary arrays...")
    codes = [load_binary_image(c) for c, m, b, s in templates]
    masks = [load_binary_image(m) for c, m, b, s in templates]

    # Generate master mask
    H, W = codes[0].shape
    master_mask = np.ones((H, W), dtype=np.uint8)
    
    # Create consolidated mask
    for i in range(H):
        for j in range(W):
            valid_count = sum(1 for mask in masks if mask[i, j] == 1)
            if valid_count < len(masks) * 0.5:  # Less than 50% have valid data
                master_mask[i, j] = 0

    print(f"📏 Template dimensions: {H}x{W}")
    print(f"🎭 Master mask valid bits: {np.sum(master_mask)}/{H*W} ({np.sum(master_mask)/(H*W)*100:.1f}%)")

    # Run Genetic Algorithm
    best_master, best_fitness, stats = run_iris_ga(templates, codes, masks, master_mask)

    # Save results
    output_path = os.path.join(output_dir, "ga_master_code.png")
    output_mask_path = os.path.join(output_dir, "ga_master_codemask.png")
    
    save_binary_image(best_master, output_path)
    save_binary_image(master_mask, output_mask_path)
    
    print(f"\n💾 GA Master template saved:")
    print(f"   - Code: {output_path}")
    print(f"   - Mask: {output_mask_path}")

    # Final evaluation
    matches, subject_matches = evaluate_final_results(output_path, output_mask_path, templates)
    
    print(f"\n🏆 Final Results:")
    print(f"   - Template matches: {matches}")
    print(f"   - Subject matches: {subject_matches}")
    print(f"   - Best GA fitness: {best_fitness:.4f}")

if __name__ == "__main__":
    main()