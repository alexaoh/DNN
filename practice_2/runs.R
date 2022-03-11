library(tfruns)

for (model in 1:4){
  training_run('neural_net.R', flags = c(model = model))
}

