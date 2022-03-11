library(tfruns)

# simple case
training_run('MNIST_flags.r',
             flags = c(hl1 = 200, hl2 = 100))
# loop case
for (hl1 in c(200, 256, 300)){
  training_run('MNIST_flags.r', flags = c(hl1 = hl1))
}
  
latest_run()
# simple case, show all runs
ls_runs()
# Better presentation of results
View(ls_runs())
# show selection runs
ls_runs(metric_val_accuracy > 0.94, order = metric_val_accuracy)
# compare_runs() Render a visual comparison of two training runs. 
# combine compare_runs() with ls_runs()
compare_runs(ls_runs(metric_val_accuracy> 0.978, order = metric_val_accuracy))
