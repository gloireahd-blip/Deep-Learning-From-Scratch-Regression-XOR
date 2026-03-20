

# 🧠 Deep Learning From Scratch : Régression (Linéaire & Polynomiale) & XOR

Ce dépôt présente une série d'implémentations fondamentales de l'intelligence artificielle réalisées uniquement avec **NumPy**. L'objectif est de démontrer comment construire des modèles prédictifs sans frameworks (comme TensorFlow ou PyTorch), en optimisant chaque calcul pour les environnements à ressources limitées (**4 Go de RAM**).

## 🚀 Contenu du Projet

Le fichier principal regroupe trois étapes clés de l'apprentissage automatique :

1. **Régression Linéaire** : Apprentissage de relations simples de type $y = ax + b$.
2. **Régression Polynomiale "From Scratch"** : 
   - Modélisation de relations non-linéaires complexes.
   - Apprentissage d'une courbe par l'ajustement des poids et des puissances des variables d'entrée.
3. **Classification XOR "From Scratch"** :
   - Mise en œuvre d'un réseau de neurones à couche cachée pour résoudre un problème de logique non-linéaire.
   - Visualisation des courbes d'apprentissage et de la frontière de décision.

## 🛠️ Formules Mathématiques Utilisées

### 1. Forward Propagation (Passage Avant)
Le calcul de la sortie d'un neurone suit la règle :
$$Z = X \cdot W + B$$
$$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$
*(La fonction d'activation Sigmoïde $\sigma$ est utilisée pour le réseau de neurones XOR)*

### 2. Fonction de Coût (Erreur Quadratique Moyenne)
Pour mesurer l'erreur entre la prédiction ($\hat{y}$) et la réalité ($y$) :
$$MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

### 3. Backpropagation (Mise à jour des Poids)
L'apprentissage se fait via la descente de gradient pour minimiser l'erreur :
$$W_{nouveau} = W_{ancien} - \eta \cdot \frac{\partial Loss}{\partial W}$$
*(Où $\eta$ représente le Learning Rate)*

## 📁 Fichiers
- `deepLearning_from_scratch.ipynb` : Le notebook complet contenant les trois implémentations, les explications mathématiques et les visualisations graphiques (Matplotlib).

---
*Note : Ce projet est optimisé pour fonctionner sur des configurations matérielles légères.*
