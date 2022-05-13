# Task 2 - Point 4
# Explore if a batch size of 16, 32 or 64 gives the best results.  

library(keras)
library(readr)
library(pROC)
library(caret)

glob.seed <- 1234
set.seed(glob.seed)
############## Hyperparameter Flags
FLAGS <- flags(
  flag_integer("batch_size", 16) # Set default to batch size 16. 
)

# Load the model from the Rmd file. 
new.model <- load_model_hdf5("convnet.h5")

# Redo the Image Data Generator definitions. 
img_width <- img_height <- 128
target_size <- c(img_width, img_height)
batch_size <- FLAGS$batch_size # Set the batch size to the given flag.
epochs <- 30
channels <- 1 # RGB = 3 channels
image_size <- c(target_size, channels)

# optional data augmentation.
train_data_gen <- image_data_generator(
  rescale = 1/255,
  validation_split = 1/5,
  # ,rotation_range = 40, # Not relevant for us. 
  # Images will almost always be relatively straight. 
  width_shift_range = 0.1, # Shift in x direction. 
  height_shift_range = 0.1, # Shift in y direction. 
  # shear_range = 0.2, # I do not want to use shearing either, 
  # as it seems a bit irrelevant for our type of images. 
  zoom_range = 0.2,
  # horizontal_flip = TRUE, # Deemed irrelevant. 
  fill_mode = "constant", # Added constant fill-mode (cval = 0), 
  # since the rest of the fill-modes seem to distort the images 
  # in a very unnatural way. Therefore I think it is better to 
  # simply set the points outside the boundaries of the input to 0. 
  brightness_range = c(0.7, 1.3) # Play with the brightness. 
)

# We do not apply the data-augmentation (except from rescaling)
# to the testing data set, since it is important 
# to validate on the true images we have been given. 
test_data_gen <- image_data_generator(
  rescale = 1/255
)

# Then we load the data using the generators. 
# training images.
train.image_array_gen <- flow_images_from_directory(train.path, 
                                                    train_data_gen,
                                                    class_mode = "binary",
                                                    seed = glob.seed, # Change to params$seed in Rmd.
                                                    target_size = target_size,
                                                    batch_size = batch_size,
                                                    color_mode = "grayscale",
                                                    subset = "training")

# validation images. 
val.image_array_gen <- flow_images_from_directory(train.path, 
                                                  train_data_gen,
                                                  class_mode = "binary",
                                                  seed = glob.seed, # Change to params$seed in Rmd.
                                                  target_size = target_size,
                                                  batch_size = batch_size,
                                                  color_mode = "grayscale",
                                                  subset = "validation")

test.image_array_gen <- flow_images_from_directory(test.path, 
                                                   test_data_gen,
                                                   class_mode = "binary",
                                                   seed = glob.seed, # Change to params$seed in Rmd.
                                                   target_size = target_size,
                                                   batch_size = 1,
                                                   color_mode = "grayscale",
                                                   shuffle = F # Makes it easier to check with 
                                                   # true class labels after predicting.
)

# number of training samples
train_samples <- train.image_array_gen$n
# number of validation samples
valid_samples <- val.image_array_gen$n
# number of testing samples
test_samples <- test.image_array_gen$n

cb <- callback_reduce_lr_on_plateau( # Reduce learning rate if val_loss plateaus. 
  monitor="val_accuracy",
  factor = 0.5,
  patience = 2, 
  min_lr = 10e-8, 
  mode = "max")

# Refit the model with the new batch_size
history <- new.model %>% fit( 
  train.image_array_gen,
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = val.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  callbacks = cb
)
plot(history)

# Evaluate the model. 
new.model %>% evaluate(test.image_array_gen, steps = test_samples)

cat("The batch size was ", FLAGS$batch_size)

y_pred <- new.model %>% predict(test.image_array_gen, steps = test_samples) %>% 
    `>`(0.5) %>% k_cast("int32")
y_pred <- as.array(y_pred)
confusionMatrix(as.factor(test.image_array_gen$classes), as.factor(y_pred))


## ---------------------------------------------------------------------------------
roc_sae_test <- roc(response = as.numeric(test.image_array_gen$classes), 
                    predictor = as.numeric(y_pred))
plot(roc_sae_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae"), lty = c(1), col = c("blue"))
