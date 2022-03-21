# hyperparametres options

# layers, units1, unit2, unit3, dropout, kernel_regularizer

# scenarios 

colnam <- c("layers", "units1", "units2", "units3", "dropout", "kernel_regularizer")
sce1 <- c(2, 10, 10, 0, 0,"NULL")
sce2 <- c(3, 20, 10 ,5, 0, "NULL")
sce3 <- c(2, 10, 10 ,0, 0.4, "NULL")
sce4 <- c(3, 20, 10 ,5, 0.4, "NULL")
sce5 <- c(2, 10, 10 ,0, 0.4, "regularizer_l2(l = 0.01)")
sce6 <- c(3, 20, 10 ,5, 0.4, "regularizer_l2(l = 0.01)")

scenarios <- rbind(sce1,sce2,sce3,sce4,sce5,sce6)

colnames(scenarios) <- colnam

scenarios <- as.data.frame(scenarios)

scenarios[,1:5] <- apply(scenarios[,1:5],2,as.numeric)




library(tfruns)

# simple case

training_run("practica_dnn_2_solucio_english.R")


training_run("practica_dnn_2_solucio_english.R",
             flags= c(layers = 2,
                      units1 = 10,
                      units2= 10,
                      units3= 0,
                      dropout= 0,
                      kernel_regularizer=  "NULL"
             )
)


for ( i in 1:nrow(scenarios))
#  i <- 1
  training_run("practica_dnn_2_solucio_english.R",
              flags= c(layers = scenarios$layers[i],
                       units1 = scenarios$units1[i],
                       units2=  scenarios$units2[i],
                       units3=  scenarios$units3[i],
                       dropout= scenarios$dropout[i],
                       kernel_regularizer= scenarios$kernel_regularizer[i]
                       )
  )
  

# Check the performance to different scenarios
 
performance.table <-  ls_runs(metric_val_accuracy > 0.7, order = metric_val_accuracy)


#performance.table$flag_layers[1]

# Run the best architecture to obtain the final weights
i <- 1
training_run("practica_dnn_2_solucio_english.R",
             flags= c(layers = performance.table$layers[i],
                      units1 = performance.table$units1[i],
                      units2=  performance.table$units2[i],
                      units3=  performance.table$units3[i],
                      dropout= performance.table$dropout[i],
                      kernel_regularizer= performance.table$kernel_regularizer[i],
                      finish = "si"
             )
)





latest_run()
# simple case, show all runs
ls_runs()
# Better presentation of results
View(ls_runs())


compare_runs(ls_runs(latest_n = 2))
