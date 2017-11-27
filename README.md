# Predictive State Controlled Models

This is the MATLAB code for implementing predictive state controlled models with random Fourier features (RFF-PSR).
For more details see

A. Hefny, C. Downey and G. Gordon, "An Efficient, Expressive and Local Minima-free Method for Learning Controlled Dynamical Systems", AAAI 2018.

## Running the code
In the code folder run:
```
load_paths
exp_synth
```

The script `exp_synth` runs RFF-PSR as well as a number of baselines on a synthetic dataset. Gram matrix version of HSE-PSR is disabled by default as it takes a long time for training and evaluation. To run Gram matrix HSE-PSR, set the variable `evaluate_hsepsr` in the script to 1.

The function `train_rffpsr` trains RFF-PSR given observation and action trajectories.
