"""
pso.py
------
Particle Swarm Optimization for KNN feature & hyperparameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def fitness(particle, X, y, cv=5):
    feature_mask = particle[:30] >= 0.5
    k = int(np.clip(particle[30], 3, 15))
    metric = 'euclidean' if particle[31] < 0.5 else 'manhattan'
    if not np.any(feature_mask):
        return 0
    X_sel = X[:, feature_mask]
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    return np.mean(cross_val_score(knn, X_sel, y, cv=cv, scoring='accuracy'))

def run_pso(X, y, num_particles=30, max_iter=50, w=0.5, c1=2, c2=0.5):
    dim = 32
    particles = [np.random.uniform(0, 1, dim) for _ in range(num_particles)]
    velocities = [np.zeros(dim) for _ in range(num_particles)]
    p_best, p_best_scores = particles.copy(), [fitness(p, X, y) for p in particles]
    g_best = p_best[np.argmax(p_best_scores)]
    acc_hist = []

    for t in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (p_best[i] - particles[i])
                + c2 * r2 * (g_best - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            score = fitness(particles[i], X, y)
            if score > p_best_scores[i]:
                p_best[i], p_best_scores[i] = particles[i], score
        g_best = p_best[np.argmax(p_best_scores)]
        acc_hist.append(max(p_best_scores))
        print(f"PSO | Iteration {t+1}: Best Accuracy = {max(p_best_scores):.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(acc_hist, marker='o', color='green')
    plt.title("PSO + KNN Accuracy Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    return g_best
