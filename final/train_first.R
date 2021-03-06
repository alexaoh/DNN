# NO NEED TO DELIVER THIS, AS IT IS THE SAME CODE AS IN model.R
# USED THIS ON MARKOV.

#install.packages("tidyverse")
library(tidyverse)
#install.packages("keras")
library(keras)
#install.packages("tensorflow")
library(tensorflow)
#install.packages("reticulate")
library(reticulate)

#install_tensorflow(extra_packages="pillow")
#install_keras()
#tf$debugging$set_log_device_placement(TRUE)
# 
# # As stated in the tutorial: 
# # https://forloopsandpiepkicks.wordpress.com/2021/03/16/how-to-build-your-own-image-recognition-app-with-r-part-1/
# # we use the dataset of birds from Kaggle: https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download
# # I extracted the first 50 types of birds from both the training and the test data downloaded. 
# 
#setwd("/home/ajo/gitRepos/DNN/final")

global.seed <- 2022 # Use this as the seed everywhere. 
set.seed(global.seed)
path.train <- "train/"
path.test <- "test/"
label.list <- dir(path.train)
output.n <- length(label.list) # as we can see, we have 50 different birds in train images.
#length(dir(path.test)) # The same is the case for the testing images, as seen below.
#all(label.list == dir(path.test))
save(label.list, file="label_list.RData") # Save the list of names on disk for later.

width <- height <- 224 # This is the original size of the images.
target.size <- c(width, height)
rgb <- 3 # Color channels.

# Set bath size and number of epochs for training later. 
batch_size <- 32
epochs <- 6

# # Make generator for training and validation data. 
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

# Get an idea of our data.
#table(train.images$classes)
#table(validation.images$classes)
#table(test.images$classes)

# Plot image number 17.
#plot(as.raster(train.images[[1]][[1]][17,,,]))
# 
# # We use a pretrained model to get good results off the bat. 
mod.base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
#summary(mod.base)
freeze_weights(mod.base)
#summary(mod.base) # Freeze the weights of the pretrained xception model. 
# 
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
# 
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
