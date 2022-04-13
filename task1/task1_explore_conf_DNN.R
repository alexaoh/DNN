# Task 1 - Point 5
# Explore if 5, 10 or 20 nodes in the fully connected layer gives the best results in the classifier. 

library(keras)
library(readr)
library(pROC)
library(caret)

# Load data generated for this purpose in main Rmd. 
data.train <- read.csv("train_for5.csv")[,-1]
data.test <- read.csv("test_for5.csv")[,-1]
xtrain <- data.matrix(data.train[,-1])
ylabels <- data.matrix(data.train[,1])
xtest <- data.matrix(data.test[,-1])
ytestlabels <- data.matrix(data.test[,1])
percentage <- dim(xtrain)[[2]]

############## Hyperparameter Flags
FLAGS <- flags(
  flag_integer("units",5) # Set default to 5 units in the fully connected layer. 
)

# Recreate the pretrained autoencoder from the Rmd file. 
# We define the same encoders. 

input_enc1 <- layer_input(shape = percentage)
output_enc1 <- input_enc1 %>% 
  layer_dense(units=1000,activation="relu") 
encoder1 <- keras_model(input_enc1, output_enc1)

input_enc2 <- layer_input(shape = 1000)
output_enc2 <- input_enc2 %>% 
  layer_dense(units=100,activation="relu") 
encoder2 <- keras_model(input_enc2, output_enc2)

input_enc3 <- layer_input(shape = 100)
output_enc3 <- input_enc3 %>% 
  layer_dense(units=50,activation="relu") 
encoder3 <- keras_model(input_enc3, output_enc3)

# Build the same autoencoder architecture as earlier. 
sae_input1 <- layer_input(shape = percentage)
sae_output1 <- sae_input1 %>% 
  encoder1() %>% 
  encoder2() %>%
  encoder3()

new.model <- keras_model(sae_input1, sae_output1)

# Load the weights that were saved after training in Rmd. 
load_model_weights_hdf5(new.model, "sae.hdf5")

# EXTRA: Make sure that the predictions from the model in Rmd and this model are the same. Good!
new_predictions <- predict(new.model, xtest)
yhat <- read.csv("predictions.csv")
yhat <- yhat[,-1]
colnames(yhat) <- c()
all.equal(data.matrix(yhat), new_predictions, check.attributes = F)

# Add the fully connected layers to the end of the autoencoder. 
sae_output2 <- sae_output1 %>%
   layer_dense(FLAGS$units,activation = "relu") %>% # Couple with fully connected layers (DNN).
   layer_dense(1,activation = "sigmoid")

sae <- keras_model(sae_input1, sae_output2)
summary(sae)

freeze_weights(sae,from=1,to=4) # Freeze the weights (pre-training using the SAE).
summary(sae)

sae %>% compile(
  optimizer = "rmsprop",
  loss = 'binary_crossentropy',
  metric = "acc"
)

callbacks_parameters <- callback_early_stopping(
  monitor = "val_loss",
  patience = 8,
  verbose = 1,
  mode = "min",
  restore_best_weights = FALSE
)

## ---------------------------------------------------------------------------------
sae %>% fit(
  x=xtrain,
  y=ylabels,
  epochs = 120,
  batch_size=64,
  validation_split = 0.2,
  callbacks = callbacks_parameters
)


## ---------------------------------------------------------------------------------
sae %>%
  evaluate(xtest, ytestlabels)


## ---------------------------------------------------------------------------------
yhat <- predict(sae,xtest)
yhatclass<-as.factor(ifelse(yhat<0.5,0,1))
confusionMatrix(yhatclass,as.factor(c(ytestlabels)))

cat("The number of layers used were: ", FLAGS$units)


## ---------------------------------------------------------------------------------
roc_sae_test <- roc(response = as.numeric(ytestlabels), predictor = as.numeric(yhat))
plot(roc_sae_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae"), lty = c(1), col = c("blue"))
