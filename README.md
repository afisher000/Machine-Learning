# Machine Learning Repository
I created this repository to start cleaning/organizing my old machine learning projects so useful code scripts are easy to find. 

## Project Overviews
### Digit Classification

The MNIST dataset is a classical starting project for those learning about machine learning. I added my own twist by manually creating a dataset using the OpenCV library to parse images of handwritten digits. Digits are centered/scaled to a 16x16 pixel square. To increase the amount of data, I included rotations and transformations of the digits.

With minimial hyperparameter tuning, it is clear the accuracy saturates somewhere between 98.5-99%. There is also clear confusion between 3s/5s and 4s/9s. Additional training data could target these pairs. 

### Credit Defaults

This is a larger kaggle dataset with multiple tables. Efficient analysis requires developing reusable functions for aggregations and joins. It is also good to be away of memory usage and how simply changing data_types can lead to large (x4!) saving in memory.

### GUI for Dataset Exploration

The majority of data science work involved the munging, cleaning, and exploration of data. While data visualization tools already exist, I developed my own user interface for quickly analyzing kaggle datasets. The plotting tool allows you to choose X and/or Y data with optional colors/hues and generated the appropriate figure types (scatter, bar, hist, pie, etc). There is a feature engineering component where numeric values can be binned/clipped/or transformed. Categorical variables can be one-hot encoded. Finally, there is a modeling window where the features, scaling, and imputation of the data pipeline can be chosen interactively. The user can choose different models, parameters, or cross-validation param_grids for hyper-parameter tuning. Finally, the models can be saved for future comparison.

### Particle-Tracking Surrogate Model

Particle-tracking simulations (and other analytic models) are a must-have for designing experiments in particle physics. There is always a balance between accuracy and computation, and sometimes, it can be helpful to develop machine learning surrogate models that are trained on many simulations. Specifically, I simulate the longitudinal phase space (energy and z-position of particles) to understand how the beam energy spread and bunch length evolves due to charge and the RF amplitude and phase in the accelerating cavities. It can be tempting to throw neural networks at the problem, but we would enjoy a semi-understandable model (to benchmark against our physical intuition). My intuition suggests the problem should be linear to first order and in fact a linear-regression model trained with second-order polynomial terms gives an fast/accurate representation of the beam dynamics. 
