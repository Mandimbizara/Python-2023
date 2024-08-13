import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog
import pandas as pd

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------
print("")
print(f"AFFECTATION")
# Créer une matrice où chaque élément représente
# l'heure pendant laquelle les individus réalisent les tâches correspondantes
cost_matrix = np.array([[10, 9, 6, 18, 11],
                        [13, 19, 6, 12, 14],
                        [3, 2, 4, 4, 5],
                        [18, 9, 12, 17, 15],
                        [11, 6, 14, 19, 10]])

# Résoudre le problème d’affectation
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Afficher l’affectation optimale et la durée totale
print("Affectation optimale:")
for i, j in zip(row_indices, col_indices):
    print(f"La tâche {i + 1} est attribuée à l'individu {j + 1}")
total_time = cost_matrix[row_indices, col_indices].sum()
print(f"Duréé totale minimale : {total_time} heures \n")

# -----------------------------------------------------------------------------------

print(f"SIMPLEXE")
# Exemple de problème de programmation linéaire
c = [-1, -2]  # Fonction objective à minimiser : c[0]*x[0] + c[1]*x[1]
A = [[-3, 2], [-1, 2], [1, 1]]  # Coefficients des contraintes (coefficients de la matrice A)
b = [2, 4, 5]  # Valeurs des contraintes

# Bornes des variables
x0_bounds = (0, None)  # x[0] >= 0
x1_bounds = (0, None)  # x[1] >= 0

res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

# Examinez les résultats
print("Valeurs optimales des variables de décision:", res.x)
print("Valeur optimale de la fonction objectif (résultat optimal z_min):", res.fun)
# print("Succès de l'optimisation:", res.success)

# -----------------------------------------------------------------------------------
print("\nQUEL EST LA VALEUR DE SORTIE ?")
a, b, c = 5, 3, 2
d = a + b * c
e = d % 4
print(e)

print("")
print("MACHINE LEARNING")

donnees = {'A': [1, 2, 3, 4],
           'B': [5, 6, 7, 8],
           'C': [9, 10, 11, 12]}
df = pd.DataFrame(donnees)

donnees_selectionnees = df.iloc[:2, :]

print(donnees_selectionnees)
