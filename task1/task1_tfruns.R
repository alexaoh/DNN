# Hyperparameter options 
# Units: 5, 10 or 20 nodes in fully connected layer.

scenarios <- c(5, 10, 20)
library(tfruns)

for (u in scenarios){
  training_run("task1_explore_conf_DNN.R",
               flags= c(units = u))
}

# Check the performance to different scenarios
performance.table <-  ls_runs(metric_val_acc > 0.78, order = metric_val_acc)

# Print the units of each of the runs in the performance table. 
performance.table$flag_units

# Run the best architecture to obtain the final weights.
i <- 1
# training_run("task1_explore_conf_DNN.R",flags= c(units = performance.table$flag_units[i]))


# Compare the two runs with the highest metric accuracy!
compare_runs(c(performance.table[1,1], performance.table[2,1]))
