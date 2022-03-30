# Task 1 - Stacked Autoencoder for Prediction of Breast Invasive Carcinoma (BRCA) Estrogen Receptor Status

The main task is written in [task1.Rmd](task1.Rmd). Running this file generates either pdf or html output, in addition to some files that are used for `tfruns` (sae.hdf5, test_for5.csv, train_for5.csv, predictions.csv).

In order to use `tfruns`, two separate files are used; [task1_tfruns.R](task1_tfruns.R) calls [task1_explore_conf_DNN.R](task1_explore_conf_DNN.R) for each scenario.
