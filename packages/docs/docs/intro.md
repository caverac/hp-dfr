---
sidebar_position: 1
---

# Introduction

Welcome to the **Adaptive hp-DFR** documentation. This project extends the Deep Fourier Residual (DFR) method with **adaptive hp-refinement** and **goal-oriented error estimation** for solving Partial Differential Equations (PDEs) using neural networks.

## Our Research

We propose a novel extension to the Deep Fourier Residual method that addresses key limitations of existing approaches through three innovations:

### 1. Adaptive Fourier Mode Selection

Instead of using a fixed truncation of Fourier modes, we dynamically select modes based on their contribution to the residual norm. This addresses the curse of dimensionality:

- Standard DFR requires $O(N^d)$ Fourier modes in $d$ dimensions
- Our sparse tensor methods reduce this to $O(N(\log N)^{d-1})$

### 2. Hierarchical Neural Network Architecture (hp-refinement)

We use a multi-scale network combining:

- **h-refinement**: Domain partitioning with local networks
- **p-refinement**: Adaptive network depth/width

This focuses computational resolution where the solution requires it most.

### 3. Goal-Oriented Error Estimation

For quantities of interest (QoI), we compute the dual-weighted residual to focus computational effort where it affects the output most. This directly targets what matters for your application rather than minimizing a generic error measure.

## Why Is This Novel?

The original DFR method establishes that the $H^{-1}$ dual norm loss is equivalent to the $H^1$ error for well-posed problems. Our approach addresses key limitations:

- **Curse of dimensionality**: Standard DFR doesn't scale well to higher dimensions
- **Uniform refinement inefficiency**: Using the same number of modes everywhere wastes computation
- **Energy norm mismatch**: For certain PDEs (e.g., Helmholtz), $H^{-1}$ may not control the energy-norm error

This is the first integration of hp-adaptivity and goal-oriented error estimation with DFR-style dual norm losses for physics-informed learning.

## Key Features

- **Multi-backend support**: TensorFlow, JAX, and PyTorch implementations
- **Reproducible experiments**: Scripts to reproduce all results
- **Scalable infrastructure**: AWS CDK templates for running experiments at scale
- **Comprehensive documentation**: Theory explanations and API reference

## Background

This work builds upon the Deep Fourier Residual method introduced in:

> **A Deep Fourier Residual Method for solving PDEs using Neural Networks**
> Jamie M. Taylor, David Pardo, Ignacio Muga
> [arXiv:2210.14129](https://arxiv.org/abs/2210.14129)

See the [Background](/docs/paper/summary) section for a detailed summary of the original DFR method and its theoretical foundations.

## Getting Started

Ready to dive in? Check out the [Installation Guide](/docs/getting-started/installation) to set up your environment, or explore the [Theory](/docs/theory/pinns) section to understand the mathematical foundations.
