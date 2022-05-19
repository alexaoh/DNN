# Following part 1 of the link, we implement the model.

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

#install_tensorflow(extra_packages="pillow")
#install_keras()
tf$debugging$set_log_device_placement(TRUE)

# As stated in the tutorial: 
# https://forloopsandpiepkicks.wordpress.com/2021/03/16/how-to-build-your-own-image-recognition-app-with-r-part-1/
# we use the dataset of birds from Kaggle: https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download
# I extracted the first 50 types of birds from both the training and the test data downloaded. 

setwd("/home/ajo/gitRepos/DNN/final")

global.seed <- 2022
path.train <- "train/"
path.test <- "test/"
label.list <- dir(path.train)
output.n <- length(label.list) # as we can see, we have 50 different birds in train images.
length(dir(path.test)) # The same is the case for the testing images, as seen below. 
all(label.list == dir(path.test))
save(label.list, file="label_list.RData") # Save the list of names on disk for later. 

width <- height <- 224 # This is the original size of the images. 
target.size <- c(width, height)
rgb <- 3 # Color channels.

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
                                           seed = global.seed)

# Load batches of validation data using the generator. 
validation.images <- flow_images_from_directory(path.train,
                                                train.data.gen, 
                                                subset = "validation",
                                                target_size = target.size,
                                                class_mode = "categorical",
                                                classes = label.list,
                                                seed = global.seed)

# Make generator for testing data. 
test.data.gen <- image_data_generator(rescale = 1/255)

# Load batches of testing data using the generator. 
test_images <- flow_images_from_directory(path.test,
                                          test.data.gen,
                                          target_size = target.size,
                                          class_mode = "categorical",
                                          classes = label.list,
                                          shuffle = F,
                                          seed = global.seed)

# Get an idea of our data. 
table(train.images$classes)
# Plot image number 17. 
plot(as.raster(train.images[[1]][[1]][17,,,]))

# We use a pretrained model to get good results off the bat. 
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

model <- model.function()
summary(model)

# We train the model. 
batch_size <- 32
epochs <- 6

hist <- model %>% fit(
  train.images,
  steps_per_epoch = train.images$n %/% batch_size,# Integer division. 
  epochs = epochs, 
  validation_data = validation.images,
  validation_steps = validation.images$n %/% batch_size # Integer division.
)

# We save the model after fitting, since it took a little while!
model %>% save_model_hdf5("finetunedXception1.h5")

# We evaluate our model on the test data. 
model %>% evaluate(test.images, 
                             steps = test.images$n)

# Next we test with another image. The image is taken from the 
# "images to test" directory downloaded from Kaggle. 
test.image <- image_load("5.jpg", 
                         target_size = target.size)

# We make an overview of the model's predictions. 
x <- image_to_array(test.image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred <- model %>% predict(x)
pred <- data.frame("Bird" = label_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:5,]
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred

# Next we have a look at which birds are well identified vs. 
# which birds are not so well idfentified by the model. 
# NEED TO READ THIS THOROUGHLY IN THE GUIDE (CA I MIDTEN AV SIDEN TIL PART 1!)
predictions <- model %>% 
  predict(
    generator = test.images,
    steps = test.images$n
  ) %>% as.data.frame

names(predictions) <- paste0("Class",0:49)

predictions$predicted_class <- 
  paste0("Class",apply(predictions,1,which.max)-1)
predictions$true_class <- paste0("Class",test.images$classes)

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

# Next we tune the model in order to increase its performance. 
# Instead of doing it the exact same way as shown in the blog, I use tfruns for this. 
