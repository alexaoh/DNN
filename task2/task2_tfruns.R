# Task 2 - Point 4
# Hyperparameter Tuning 
# Explore if a batch size of 16, 32 or 64 gives the best results. 


scenarios <- c(16, 32, 64)
library(tfruns)

for (u in scenarios){
  training_run("task2_explore.R",
               flags= c(batch_size = u))
}

# Check the performance to different scenarios
performance.table <-  ls_runs(metric_val_accuracy > 0.4, order = metric_val_accuracy)

# Print the units of each of the runs in the performance table. 
performance.table$flag_batch_size

# Compare the two runs with the highest metric accuracy!
compare_runs(c(performance.table[1,1], performance.table[2,1]))
