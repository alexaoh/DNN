## ----setup, include=FALSE--------------------------------------------------------------------------------------------------------
rm(list = ls())
knitr::opts_chunk$set(echo = T, warning = F, comment = "#>", size = "scriptsize")
setwd("/home/ajo/gitRepos/DNN/final")
library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
library(tfruns)


## --------------------------------------------------------------------------------------------------------------------------------
global.seed <- 2022 # Use this as the seed everywhere. 
set.seed(global.seed)
path.train <- "train/"
path.test <- "test/"
label.list <- dir(path.train) # List the bird species names in the train folder. 
output.n <- length(label.list) # As we can see, we have 50 different birds in train images.
length(dir(path.test)) # The same is the case for the testing images, as seen below.
all(label.list == dir(path.test))
save(label.list, file="label_list.RData") # Save the list of names on disk for later.

width <- height <- 224 # This is the original size of the images.
target.size <- c(width, height)
rgb <- 3 # Color channels.

# Set batch size and number of epochs for training later. 
batch_size <- 32
epochs <- 6


## --------------------------------------------------------------------------------------------------------------------------------
# Make generator for training and validation data. 
train.data.gen <- image_data_generator(rescale = 1/255,
                                       validation_split = 0.2)

# Load batches of training data using the generator.
train.images <- flow_images_from_directory(path.train,
                                           train.data.gen,
                                           subset = "training",
                                           target_size = target.size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label.list,
                                           seed = global.seed, 
                                           batch_size = batch_size)

# Load batches of validation data using the generator.
validation.images <- flow_images_from_directory(path.train,
                                                train.data.gen,
                                                subset = "validation",
                                                target_size = target.size,
                                                class_mode = "categorical",
                                                classes = label.list,
                                                seed = global.seed, 
                                                batch_size = batch_size)

# Make generator for testing data. Do not want the validation_split here. 
test.data.gen <- image_data_generator(rescale = 1/255)

# Load batches of testing data using the generator.
test.images <- flow_images_from_directory(path.test,
                                          test.data.gen,
                                          target_size = target.size,
                                          class_mode = "categorical",
                                          classes = label.list,
                                          shuffle = F,
                                          seed = global.seed, 
                                          batch_size = 1) # Set batch size to 1 in order to
                                                          # test on one image at a time. 


## --------------------------------------------------------------------------------------------------------------------------------
# Get an idea of our data.
table(train.images$classes)
table(validation.images$classes)
table(test.images$classes)

# Plot image number 17.
plot(as.raster(train.images[[1]][[1]][17,,,]))


## --------------------------------------------------------------------------------------------------------------------------------
# We use the pretrained model to get good results off the bat. 
mod.base <- application_xception(weights = 'imagenet', 
                                  include_top = FALSE, input_shape = c(width, height, 3))
summary(mod.base)
freeze_weights(mod.base)
summary(mod.base) # Freeze the weights of the pretrained xception model. 

model.function <- function(learning_rate = 0.001, 
                           dropoutrate=0.2, n_dense=1024){
  # Function to add layers to the pre-trained model. This is finetuned.
  k_clear_session()

  model <- keras_model_sequential() %>%
    mod.base %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output.n, activation="softmax")

  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  return(model)
}
# 
model <- model.function()
#summary(model)

# We train the model.
# This training is left for the larger computer, via ssh.
hist <- model %>% fit(
  train.images,
  steps_per_epoch = train.images$n %/% batch_size,# Integer division.
  epochs = epochs,
  validation_data = validation.images,
  validation_steps = validation.images$n %/% batch_size # Integer division.
)

# We save the model after fitting, since it took a little while!
model %>% save_model_hdf5("finetunedXception1.h5")


## --------------------------------------------------------------------------------------------------------------------------------
# Load the model fitted on the other computer. 
model <- load_model_hdf5("finetunedXception1.h5")

# We evaluate our model on the test data. 
model %>% evaluate(test.images, 
                             steps = test.images$n)
# 84% accuracy with first model. 


## --------------------------------------------------------------------------------------------------------------------------------
# Next we test with another image. The image is taken from the 
# Wikipedia page on Abbot's babbler. 
test.image <- image_load("abbotsBabbler.jpg", 
                         target_size = target.size)

# We make an overview of the model's predictions. 
x <- image_to_array(test.image)
x <- array_reshape(x, c(1, dim(x))) # Reshape image to expected input dimensions by model. 
x <- x/255 # Rescale pixel values. 
plot(as.raster(x[1,,,])) # Plot the image. 
pred <- model %>% predict(x) # Make predictions on image. 
# Make dataframe of predictions, with names of the respective birds. 
pred <- data.frame("Bird" = label.list, "Probability" = t(pred))
# Order the probabilities decreasingly and only show the largest 5. 
pred <- pred[order(pred$Probability, decreasing=T),][1:5,] 
# Change the probability to percentage.
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred 
# 72% probability. 
# This means that the model would have classified this test image correctly. 


## --------------------------------------------------------------------------------------------------------------------------------
# Next we have a look at which birds are well identified vs. 
# which birds are not so well identified by the model. 

# We predict on the testing images and save the predictions as a data frame. 
predictions <- model %>% 
  predict(
    test.images,
    steps = test.images$n
  ) %>% as.data.frame
# Each testing image is given in each row, where each column value is the probability
# of the image belonging to that species of bird, according to the model. 

# Change the column names to "Class<i>" respectively, i \in \{0,49\}.
names(predictions) <- paste0("Class",0:49)

# Add another column to the dataframe which tells us which class has the highest 
# probability. Thus, this is the class that the model would predict. 
predictions$predicted_class <- 
  paste0("Class",apply(predictions,1,which.max)-1)
# Add the true classes to the dataframe as well. 
predictions$true_class <- paste0("Class",test.images$classes)


## ---- fig.height=8---------------------------------------------------------------------------------------------------------------
# Finally, we count the percentage of correct classifications 
# (since each of our classes has exactly 5 test images, 
# the values will be either 0, 20, 40, 60, 80 or 100%). 
predictions %>% group_by(true_class) %>% 
  summarise(percentage_true = 100*sum(predicted_class == 
                                        true_class)/n()) %>% 
  left_join(data.frame(bird= names(test.images$class_indices), 
                       true_class=paste0("Class",0:49)),by="true_class") %>%
  select(bird, percentage_true) %>% 
  mutate(bird = fct_reorder(bird,percentage_true)) %>%
  ggplot(aes(x=bird,y=percentage_true,fill=percentage_true, 
             label=percentage_true)) +
  geom_col() + theme_minimal() + coord_flip() +
  geom_text(nudge_y = 3) + 
  ggtitle("Percentage correct classifications by bird species")


## --------------------------------------------------------------------------------------------------------------------------------
# Next we tune the model in order to increase its performance. 
# Instead of doing it the exact same way as shown in the blog, I use tfruns for this. 
# This is done separately in the files "tfruns.R" and "finetuning.R", where
# the former calls on the latter for ever scenario and is used to compare runs in the end. 


## --------------------------------------------------------------------------------------------------------------------------------
knitr::kable(c(0.001,0.0001), caption = "Values of Learning Rate")
knitr::kable(c(0.3,0.2), caption = "Values of Dropout Rate")
knitr::kable(c(1024,256), caption = "Values of Number of Dense Nodes")


## --------------------------------------------------------------------------------------------------------------------------------
# Read the performance table produced by the remote computer. 
performance.table <- read.csv("performance_table.csv")

# Compare the two runs with the highest validation accuracy.
#compare_runs(c(performance.table[1,2], performance.table[2,2]))
# This comparison cannot be done when compiling the Rmd. 
# However, we can see what the values of the two models with 
# best validation accuracy are below. 
df.tf <- cbind(performance.table$metric_val_accuracy, 
               performance.table$flag_dropout_rate,
               performance.table$flag_learning_rate, 
               performance.table$flag_n_dense)
colnames(df.tf) <- c("Validation accuracy", "Dropout rate", "Learning rate", "Dense nodes")
knitr::kable(df.tf, caption = "Validation Accuracy and Hyperparameter Values of all Trained Models")
# It is apparent that the model with the best validation accuracy has
# dropout rate 0.2, learning rate 1e-04 and n_dense 1024. 
# We train this model again on the remote computer and use this as the final model. 


## --------------------------------------------------------------------------------------------------------------------------------
# Load the model fitted on the other computer. 
mod <- load_model_hdf5("tunedHypParamXception.h5")

# We evaluate our model on the test data. 
mod %>% evaluate(test.images, 
                             steps = test.images$n)


## --------------------------------------------------------------------------------------------------------------------------------
# Next we test with another image. The image is taken from the 
# Wikipedia page on Abbot's babbler. 
test.image <- image_load("abbotsBabbler.jpg", 
                         target_size = target.size)

# We make an overview of the model's predictions. 
x <- image_to_array(test.image)
x <- array_reshape(x, c(1, dim(x))) # Reshape image to expected input dimensions by model. 
x <- x/255 # Rescale pixel values. 
plot(as.raster(x[1,,,])) # Plot the image. 
pred <- model %>% predict(x) # Make predictions on image. 
# Make dataframe of predictions, with names of the respective birds. 
pred <- data.frame("Bird" = label.list, "Probability" = t(pred))
# Order the probabilities decreasingly and only show the largest 5. 
pred <- pred[order(pred$Probability, decreasing=T),][1:5,] 
# Change the probability to percentage.
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred 
# We can see that the first model gives Abbot's babbler an 65% probability. 
# This means that the model would have classified this test image correctly. 


## --------------------------------------------------------------------------------------------------------------------------------
# Next we have a look at which birds are well identified vs. 
# which birds are not so well identified by the model. 

# We predict on the testing images and save the predictions as a data frame. 
predictions <- mod %>% 
  predict(
    test.images,
    steps = test.images$n
  ) %>% as.data.frame
# Each testing image is given in each row, where each column value is the probability
# of the image belonging to that species of bird, according to the model. 

# Change the column names to "Class<i>" respectively, i \in \{0,49\}.
names(predictions) <- paste0("Class",0:49)

# Add another column to the dataframe which tells us which class has the highest 
# probability. Thus, this is the class that the model would predict. 
predictions$predicted_class <- 
  paste0("Class",apply(predictions,1,which.max)-1)
# Add the true classes to the dataframe as well. 
predictions$true_class <- paste0("Class",test.images$classes)


## ---- fig.height=8---------------------------------------------------------------------------------------------------------------
# Finally, we count the percentage of correct classifications 
# (since each of our class has exactly 5 test images, 
# the values will be either 0, 20, 40, 60, 80 or 100%). 
predictions %>% group_by(true_class) %>% 
  summarise(percentage_true = 100*sum(predicted_class == 
                                        true_class)/n()) %>% 
  left_join(data.frame(bird= names(test.images$class_indices), 
                       true_class=paste0("Class",0:49)),by="true_class") %>%
  select(bird, percentage_true) %>% 
  mutate(bird = fct_reorder(bird,percentage_true)) %>%
  ggplot(aes(x=bird,y=percentage_true,fill=percentage_true, 
             label=percentage_true)) +
  geom_col() + theme_minimal() + coord_flip() +
  geom_text(nudge_y = 3) + 
  ggtitle("Percentage correct classifications by bird species")
