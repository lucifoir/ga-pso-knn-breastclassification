"""
ga.py
-----
Genetic Algorithm implementation for simultaneous feature selection
and KNN hyperparameter optimization.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def fitness(individual, X, y, cv=5):
    feature_mask = individual[:30] >= 0.5
    k = int(np.clip(individual[30], 3, 15))
    metric = 'euclidean' if individual[31] < 0.5 else 'manhattan'

    if not np.any(feature_mask):
        return 0
    X_sel = X[:, feature_mask]
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    return np.mean(cross_val_score(knn, X_sel, y, cv=cv, scoring='accuracy'))

def init_population(pop_size, dim):
    return [np.random.uniform(0, 1, dim) for _ in range(pop_size)]

def selection(pop, fit_scores, k=3):
    selected = random.sample(list(zip(pop, fit_scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def mutate(ind, rate=0.1):
    for i in range(len(ind)):
        if random.random() < rate:
            ind[i] += np.random.normal(0, 0.1)
            ind[i] = np.clip(ind[i], 0, 1)
    return ind

def run_ga(X, y, pop_size=30, generations=50):
    dim = 32
    population = init_population(pop_size, dim)
    acc_hist = []

    for gen in range(generations):
        fitnesses = [fitness(ind, X, y) for ind in population]
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = selection(population, fitnesses), selection(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop
        best = max(fitnesses)
        acc_hist.append(best)
        print(f"GA | Generation {gen+1}: Best Accuracy = {best:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(acc_hist, marker='o', color='blue')
    plt.title("GA + KNN Accuracy Progress")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    best_idx = np.argmax([fitness(ind, X, y) for ind in population])
    return population[best_idx]
