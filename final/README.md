# Final Project

The image recognition application for 50 species of birds is published [here](https://alexaoh.shinyapps.io/birdapp/). A final [report](report.pdf) of the entire process, as well as the code, can be found in this repo. This file quickly highlights what the different files and directories in this directory contain. 

* [report.pdf](report.pdf): final report in pdf format. 
* [report.html](report.html): final report in html format. 
* [report.Rmd](report.Rnd): source code in Rmd fornat for final report. 
* [full_code.R](full_code.R): source code in R format for final report. Generated using knitr::purl from report.Rmd. 
* [birdapp](birdapp): files for Shiny app. 
* [train](train): training data. 
* [test](test): testing data. 
* [tfruns.R](tfruns.R): code used for hyperparameter tuning. 
* [finetuning.R](finetuning.R): code used for hyperparameter tuning. 
* [performance_table.csv](performance_table.csv): resulting performance table from finetuning with tfruns. 
* [runs](runs): resulting runs from finetuning with tfruns. 
* [finetunedXception1.h5](finetunedXception1.h5): saved model image from training of first model. 
* [tunedHypParamXception.h5](tunedHypParamXception.h5): saved model image from training of second/new model. 
* [label_list.RData](label_list.RData): list of labels of bird species saved from compilation of Rmd. 
* [abbotsBabbler.jpg](abbotsBabbler.jpg): Wikipedia image of Abbot's babbler.   
