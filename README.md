# Prediction-Phase-Class-Microemulsion-Formation-Emultant-Compund-Using-ML-Algorithm-with-FI

This repository contains the implementation of machine learning models to predict the phase class of microemulsion formation based on the composition of emultant compounds.
The study applies several classification algorithms and feature importance analysis to identify the most influential factors affecting microemulsion phase formation.

This project was developed as part of research on microemulsion formulation prediction using machine learning techniques.

Overview

Microemulsion is a thermodynamically stable colloidal system consisting of oil, water, surfactant, and co-surfactant. Determining the phase class of microemulsion normally requires laboratory experiments which are time-consuming, costly, and resource-intensive.

To address this problem, this research proposes machine learning models to predict the phase class of microemulsion formation based on the composition of emultant compounds. 

Blind_review Prediction of Phas…

The system performs:

Data preprocessing

Model training using multiple machine learning algorithms

Model evaluation with classification metrics

Feature importance analysis

Model comparison and selection of the best model

Dataset

The dataset used in this study consists of 217 samples obtained from laboratory experiments of microemulsion formulations. 

Blind_review Prediction of Phas…

Class Distribution
Phase	Samples
Phase 1	110
Phase 2	75
Phase 3	32
Features

The dataset contains 7 numerical features describing the composition of the emulsion:

Feature	Description
Oil (mPa.s)	Oil viscosity value
Oil Amount (g)	Amount of oil
Surfactant (HLB)	Hydrophilic-Lipophilic Balance value
Surfactant Amount (g)	Amount of surfactant
Water Phase (V)	Water phase volume
Water Phase Amount (g)	Amount of water phase
Co-Surfactant Ratio	Ratio of co-surfactant to surfactant

Target variable:

Phase → Microemulsion phase class (1, 2, or 3)

Machine Learning Models

Four classification algorithms are implemented:

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes

Decision Tree

For SVM, four kernels are tested:

Linear

Polynomial

Radial Basis Function (RBF)

Sigmoid

The training pipeline also includes:

StandardScaler

SMOTE (to handle class imbalance)

Stratified train-test split

K-Fold cross validation

These models are implemented in the training script. 

training

Workflow

The machine learning pipeline consists of the following steps:

Load dataset

Data cleaning

Handling missing or invalid values

Data preprocessing and normalization

Train-test split (80% training, 20% testing)

Model training

Model evaluation

Feature importance analysis

Model comparison

Best model selection and saving

Feature Importance Analysis

To identify the most influential variables, several methods are used:

Mutual Information

LinearSVC (L1 and L2 regularization)

Permutation Importance

SHAP (SHapley Additive Explanations)

Feature importance analysis helps determine which variables most influence the formation of microemulsion phases.

The most important features identified include:

Surfactant

Oil

Water Phase

Surfactant Amount

Co-Surfactant

Oil Amount

Evaluation Metrics

Model performance is evaluated using:

Accuracy

Precision

Recall

F1-Score

These metrics are calculated using macro averaging to handle the multi-class classification problem. 

Blind_review Prediction of Phas…

Cross Validation

To ensure model stability, 5-Fold Cross Validation is used during training.

This approach ensures that:

The model generalizes well

Overfitting is minimized

Each fold maintains class distribution using stratified sampling.
