# Task 2 - Point 4
# Hyperparameter Tuning 
# Explore learning rate, dropout rate and number of nodes in the final dense hidden layer. 

# library(tidyverse)
# library(keras)
# library(tensorflow)
# library(reticulate)
# library(tfruns)
# 
# tune_grid <- data.frame("learning_rate" = c(0.001,0.0001),
#                         "dropoutrate" = c(0.3,0.2),
#                         "n_dense" = c(1024,256))
# lr <- c(0.001,0.0001)
# dr <- c(0.3,0.2)
# n.dense <- c(1024,256)
# scenarios <- expand.grid(lr, dr, n.dense)
# 
# for (i in 1:8){
#   training_run("finetuning.R",
#                flags= c(learning_rate = scenarios[i, 1], 
#                         dropout_rate = scenarios[i, 2],
#                         n_dense = scenarios[i, 3]))
# }
# 
# # Check the performance of different scenarios. Order by validation accuracy. 
# performance.table <-  ls_runs(order = metric_val_accuracy)
# 
# # Save the performance table to disk. Then it will be loaded again after 
# # completing all the runs on the remote computer. 
# write.csv(performance.table,"performance_table.csv")

performance.table <- read.csv("performance_table.csv")
# Print the units of each of the runs in the performance table. 
#performance.table$flag_ ...

# Compare the two runs with the highest metric accuracy!
#compare_runs(c(performance.table[1,1], performance.table[2,1]))
