"""
hybrid.py
---------
Sequential hybrid combining GA (Feature Selection)
and PSO (Hyperparameter Tuning) for KNN.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from .ga import init_population, selection, crossover, mutate

def run_hybrid_ga_pso(X, y, pop_size=30, generations=30, num_particles=30, max_iter=30):
    """
    Step 1: GA performs feature selection.
    Step 2: PSO tunes (k, metric) on selected subset.
    Returns: (feature_mask, best_k, metric)
    """

    # --- Step 1: GA for Feature Selection ---
    dim = 30
    population = init_population(pop_size, dim)
    acc_hist = []

    for gen in range(generations):
        fitnesses = []
        for ind in population:
            mask = ind >= 0.5
            if not np.any(mask):
                fitnesses.append(0)
                continue
            X_sel = X[:, mask]
            knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            knn.fit(X_sel, y)
            y_pred = knn.predict(X_sel)
            fitnesses.append(accuracy_score(y, y_pred))
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = selection(population, fitnesses), selection(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop
        acc_hist.append(max(fitnesses))
        print(f"GA-FS | Gen {gen+1}: Best = {max(fitnesses):.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(acc_hist, color="purple", marker="o")
    plt.title("GA Feature Selection Progress")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    best_mask = population[np.argmax(fitnesses)] >= 0.5

    # --- Step 2: PSO for Hyperparameter Optimization ---
    dim_hp = 2  # [k, metric]
    particles = [np.random.uniform(0, 1, dim_hp) for _ in range(num_particles)]
    velocities = [np.zeros(dim_hp) for _ in range(num_particles)]
    p_best = particles.copy()
    p_best_scores = []

    for p in p_best:
        k = int(np.clip(p[0]*15, 3, 15))
        metric = 'euclidean' if p[1] < 0.5 else 'manhattan'
        X_sel = X[:, best_mask]
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_sel, y)
        y_pred = knn.predict(X_sel)
        p_best_scores.append(accuracy_score(y, y_pred))

    g_best = p_best[np.argmax(p_best_scores)]
    acc_hp = []

    for t in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                0.5 * velocities[i]
                + 2 * r1 * (p_best[i] - particles[i])
                + 0.5 * r2 * (g_best - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            k = int(np.clip(particles[i][0]*15, 3, 15))
            metric = 'euclidean' if particles[i][1] < 0.5 else 'manhattan'
            X_sel = X[:, best_mask]
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_sel, y)
            y_pred = knn.predict(X_sel)
            score = accuracy_score(y, y_pred)
            if score > p_best_scores[i]:
                p_best[i], p_best_scores[i] = particles[i], score
        g_best = p_best[np.argmax(p_best_scores)]
        acc_hp.append(max(p_best_scores))
        print(f"PSO-HP | Iter {t+1}: Best = {max(p_best_scores):.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(acc_hp, color="orange", marker="o")
    plt.title("PSO Hyperparameter Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    best_hp = g_best
    best_k = int(np.clip(best_hp[0]*15, 3, 15))
    best_metric = 'euclidean' if best_hp[1] < 0.5 else 'manhattan'
    print(f"Hybrid GA+PSO | Best k={best_k}, Metric={best_metric}")
    return best_mask, best_k, best_metric
