# 🧠 Deep Learning From Scratch: Regression (Linear & Polynomial) & XOR

This repository features a series of fundamental artificial intelligence implementations built entirely from first principles using **NumPy**. The primary objective is to demonstrate how to construct predictive models without high-level frameworks (such as TensorFlow or PyTorch), optimizing mathematical operations specifically for resource-constrained environments (**4 GB RAM**).

## 🚀 Project Overview

The core implementation covers three critical evolutionary stages of machine learning:

1. **Linear Regression**: Learning basic linear relationships of the form $y = ax + b$.
2. **Polynomial Regression "From Scratch"**: 
   - Modeling complex, non-linear patterns.
   - Forcing a linear engine to fit curved boundaries by explicitly engineering higher-degree input feature matrices.
3. **XOR Classification "From Scratch"**:
   - Building a multi-layer perceptron (MLP) with a hidden layer to solve the classic non-linearly separable logic problem.
   - Dynamic live plotting of the loss landscape minimization and the resulting decision boundary.

## 🛠️ Core Mathematical Mechanics

### 1. Forward Propagation

The transformation of input features through the neural layers follows standard matrix operations:

$$Z = X \cdot W + B$$

$$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$

*(The Sigmoid activation function $\sigma$ is deployed to introduce non-linearity within the XOR network architecture).*

### 2. Loss Function (Mean Squared Error)

To quantify the divergence between the network predictions ( $\hat{y}$ ) and empirical ground truth ( $y$ ):

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

### 3. Backpropagation & Parameter Optimization

Network learning is driven by computing partial derivatives via the chain rule, updating system parameters down the error gradient:

$$W_{new} = W_{old} - \eta \cdot \frac{\partial \text{Loss}}{\partial W}$$

*(Where  $\eta$  represents the learning rate optimization step-size).*

---

## 📊 Model Optimization & Convergence Dynamics

<p align="center">
  <img src="https://github.com/user-attachments/assets/34591e0b-eb26-4fd2-8eb2-1c732bc0663b" alt="Linear Regression Live Animation" width="50%" />
  <img src="https://github.com/user-attachments/assets/a8e71a1f-5c51-4130-99bd-d67ce7a0378c" alt="Polynomial Regression Curve Fitting" width="50%" />
  <img src="https://github.com/user-attachments/assets/1ce493f2-f66e-4a56-866e-f30d1b7628e1" alt="XOR Neural Network Decision Boundary" width="50%" />
</p>

*Figure: Comparative visualization of training execution phases — Linear Regression Trend Column (1), Polynomial tracking optimization (2), and the non-linear decision landscape mapping out the XOR logical criteria (3).*

---

## 📁 Repository Structure
- `deepLearning_from_scratch.ipynb`: A comprehensive standalone Jupyter Notebook containing all vectorised NumPy implementations, step-by-step calculus layouts, and interactive Matplotlib visualization wrappers.

---
*Note: Computational graphs and matrix shapes have been structurally flattened to guarantee high performance on lightweight edge hardware configurations.*
