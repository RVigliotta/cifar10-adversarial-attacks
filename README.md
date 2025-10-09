![GitHub commit activity](https://img.shields.io/github/commit-activity/t/RVigliotta/cifar10-adversarial-attacks)
![GitHub last commit](https://img.shields.io/github/last-commit/RVigliotta/cifar10-adversarial-attacks)
![GitHub Repo stars](https://img.shields.io/github/stars/RVigliotta/cifar10-adversarial-attacks)

📁 Data          → Contains CIFAR-10 raw data and preprocessed tensors;

📁 Docs          → Contains the plots and documentation of the projects;

📁 Models        → Contains model definitions and saved checkpoints;

📁 Src           → Contains implementation of attacks, model training, evaluation and data preprocessing;


## Table of Contents 📋
* [General Information](#General-Information-ℹ)
  + [Project Goal](#Project-Goal-)
* [Features](#features-)
* [Technologies used](#Technologies-used-)
* [Experimental Results](#Experimental-Results-)
* [Improvements](#Improvements-)
* [Project Status](#project-status-)

## General Information ℹ
This project explores adversarial machine learning on CIFAR-10, investigating how neural networks can be made robust against adversarial attacks. The implementation includes:
- Fast Gradient Sign Method (FGSM) attacks
- Adversarial training techniques
- Robustness evaluation framework
- Comparative analysis of model performance

### Project Goal 🎯
The primary objectives are:
1. Demonstrate vulnerability of standard CNNs to adversarial attacks
2. Implement adversarial training defense mechanism
3. Quantify robustness-accuracy tradeoff
4. Provide visual comparison of model decision boundaries

## Features 📝
- **Adversarial Attack Implementation**: FGSM attack with customizable ε parameter
- **Robust Training Pipeline**: Hybrid training with 50% clean/50% adversarial examples
- **Evaluation Framework**:
  - Clean accuracy evaluation
  - Adversarial robustness metrics
  - Side-by-side performance comparison
- **Visualization Tools**:
  - Robustness comparison plots

## Technologies used 📊
- Python
- Numpy
- Matplot
- Torch
- Torchvision

## Experimental Results
| Model Type                   | Clean Accuracy | FGSM Accuracy (ε=0.03) |
|------------------------------|----------------|------------------------|
| Standard CNN                 | 85.42%         | 11.24%                 |
| Adversarially Trained (FGSM) | 71.94%         | 26.79%                 |


## Improvements
Potential enhancements under consideration:
- Add support for ensemble adversarial training
- Include robustness curves across ε values
- Include more complex models to improve accuracy in both clean and adversarial versions
  
## Project Status
The Project is: **_In_Progress_**. 🚧
