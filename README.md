# Optimizer Benchmark: CIFAR-10 Analysis

This project provides a systematic evaluation of stochastic optimization algorithms (SGD, RMSProp, AdaGrad, and Adam) using a Convolutional Neural Network (CNN) on the CIFAR-10 image classification task. The study focuses on how different adaptive and non-adaptive mechanisms influence convergence speed and final generalization performance.

## Project Overview

The goal of this analysis was to move beyond "plug-and-play" optimization and understand the fundamental trade-offs between different update rules. By performing a grid search over learning rates and momentum values, this project highlights why faster convergence doesn't always lead to better generalization.

## Key Findings

The experiments revealed a clear distinction between adaptive methods and traditional SGD:

| Optimizer | Best Configuration | Final Val Acc | Key Trait |
|-----------|-------------------|---------------|-----------|
| SGD | $lr=0.1$, $mom=0.0$ | 76.80% | Best Generalization |
| Adam | $lr=0.001$ | 75.94% | Fastest early convergence, but overfits |
| AdaGrad | $lr=0.01$ | 75.55% | Lowest validation loss (0.793) |
| RMSProp | $lr=0.001$ | 74.22% | Rapid initial progress, high instability at $lr > 0.001$ |

## Core Insights

- **Generalization vs. Speed:** While Adam and RMSProp converged faster in the first 5-7 epochs, they ultimately overfit the training data, leading to a larger gap between training and validation accuracy.

- **The Power of Vanilla SGD:** Despite its simplicity, SGD with a high learning rate ($lr=0.1$) achieved the highest validation accuracy (76.80%), suggesting that non-adaptive methods may find more generalizable minima in this architecture.

- **AdaGrad's Stability:** AdaGrad achieved the lowest final validation loss. However, its monotonically decreasing learning rate caused a premature plateau by epoch 12.

## Model Architecture & Setup

- **Architecture:** A 3-layer Convolutional Neural Network (32, 64, and 128 filters) followed by two fully connected layers.
- **Dataset:** CIFAR-10 (60,000 images, 10 classes).
- **Hardware:** Experiments were performed using CUDA acceleration.
- **Hyperparameter Search:** Grid search across learning rates {0.001, 0.01, 0.1} and momentum {0.0, 0.5, 0.9}.

## Visualizations

Example of the best-performing configuration (SGD, $lr=0.1$, $mom=0.0$):
<img width="610" height="286" alt="image" src="https://github.com/user-attachments/assets/39b8acc0-f342-4480-839e-2ada2b8ac70e" />

