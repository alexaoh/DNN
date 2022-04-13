# Task 1 - Stacked Autoencoder for Prediction of Breast Invasive Carcinoma (BRCA) Estrogen Receptor Status

The main task is written in [task1.Rmd](task1.Rmd). Running this file generates either pdf or html output, in addition to some files that are used for `tfruns` (sae.hdf5, test_for5.csv, train_for5.csv, predictions.csv).

In order to use `tfruns`, two separate files are used; [task1_tfruns.R](task1_tfruns.R) calls [task1_explore_conf_DNN.R](task1_explore_conf_DNN.R) for each scenario.

In order to define the concatenated model, the file [aes_practice_3.R](aes_practice_3.R) is used. It is sourced into [task1.Rmd](task1.Rmd). This is the file given by the teaching staff in the section on autoencoders, with some slight changes in naming and a bunch of code left out. Only the stacked autoencoder from this file is kept. 

The file [render.R](render.R) is used to quickly compile both html and pdf output files. Notice that the two file formats are rendered separately, meaning that the outputs in them will differ, since pseudo-random sampling is involved in the code. This could be solved by (e.g.) only rendering a html, before converting to a pdf (or vice versa), in a case where one is interested in the same report in both formats. With this in mind, we have used the results in the html-file as the basis for the discussions, in case the discussions are not 100% coherent with the results shown in the pdf. However, this should not be a problem in most cases, as the report is as dynamical as possible.

