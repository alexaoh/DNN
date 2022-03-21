
file <- "./data/parkinsons_updrs.data"
seed1 <- 1234
pr <- 2/3


library(keras)


mydata <- read.csv(file, header=TRUE)

dim(mydata)
mydata$sex <- as.factor(mydata$sex)




# Create the binary variable of Parkinson's severity

# total_UPDRS

mydata$total_sev <- ifelse(mydata$total_UPDRS > 25, 1,0)
table(mydata$total_sev)


# `Min-max` transformation


(biom_voice_mesures <- names(mydata)[7:22])

# Transformation 0- 1
# custom normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# only biom_voice_mesures normalized
mydata_nrm <- as.data.frame(lapply(mydata[,biom_voice_mesures], normalize))
summary(mydata_nrm)




boxplot(mydata_nrm, las=2, col="palegreen3", main="Normalized data")



# Separate train and test


# train/test

set.seed(seed1) #set the pseudorandom generator seed
p <- pr
n <- nrow(mydata)
      
train <- sample(n,floor(n*p))

x_train <- as.matrix(mydata_nrm[train,])
x_test <- as.matrix(mydata_nrm[-train,])

y_train <- mydata$total_sev[train]
y_test  <- mydata$total_sev[-train]

n_features <- ncol(x_train)

# Hyperparameter flags ---------------------------------------------------

# set flags for hyperparameters of interest (we include default values)
FLAGS <- flags(
  
  flag_integer("layers",2),
  flag_integer("units1", 10),
  flag_integer("units2", 10),
  flag_integer("units3", 0),
  
  flag_string("kernel_regularizer", "NULL"),
  flag_numeric("dropout",0),
  flag_string("finish", "no")
              )

# Buiding and training a dense neural network


# Define Model

# Create a model with input layer + two hidden layers  

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = FLAGS$units1, activation = "relu", input_shape = c(n_features),
                kernel_regularizer = eval(parse(text=FLAGS$kernel_regularizer))) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = FLAGS$units2, activation = "relu",
            kernel_regularizer = eval(parse(text=FLAGS$kernel_regularizer))) %>%
  layer_dropout(rate = FLAGS$dropout)


if (FLAGS$layers == 3) {
     model %>% 
      layer_dense(units = FLAGS$units3, activation = "relu",
                  kernel_regularizer = eval(parse(text=FLAGS$kernel_regularizer))) %>%
      layer_dropout(rate = FLAGS$dropout)
 
} 

  
  
  
# Add final output layer
model %>% layer_dense(units = 1, activation = "sigmoid")



summary(model)




  

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


p_validation_split = 0.3
p_callbacks = NULL


if (FLAGS$finish == "si") {
  
  # callbacks sentences
  cb.list <- list(
    callback_model_checkpoint(
      filepath = "mymodel.h5",
      monitor = "accuracy",
      save_best_only = T
    )
  ) 
  p_validation_split = 0
  p_callbacks = cb.list
  
  
}


#eval(parse(text=p_callbacks))

history <- model %>% fit(
  x_train, y_train, 
  epochs = 100, 
  callbacks = p_callbacks,
  batch_size = 32,
  validation_split = p_validation_split
)


plot(history)



# Model performance 


evaluate(model, x_test, y_test)



