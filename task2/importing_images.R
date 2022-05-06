library(keras)

img_width <- 128 # or 256, 128, 64 or 32 (try different sizes)!
img_height <- 128
target_size <- c(img_width, img_height)
batch_size <- 32
epochs <- 30

# RGB = 3 channels
channels <- 1
image_size <- c(target_size, channels)

# optional data augmentation
train_data_gen <- image_data_generator(
  rescale = 1/255, 
  validation_split = 1/3,
  # rotation_range = 40,
  # width_shift_range = 0.2,
  # height_shift_range = 0.2,
  # shear_range = 0.2,
  # zoom_range = 0.2,
  # horizontal_flip = TRUE,
  # fill_mode = "nearest"
)

# Then we load the data using the generators. 
# training images.
train.image_array_gen <- flow_images_from_directory("images", 
                                                    train_data_gen,
                                                    class_mode = "binary",
                                                    seed = 1234, # Change to params$seed in Rmd.
                                                    target_size = target_size,
                                                    batch_size = batch_size,
                                                    color_mode = "grayscale",
                                                    subset = "training")

# testing (or possibly validation) images. 
val.image_array_gen <- flow_images_from_directory("images", 
                                                   train_data_gen,
                                                   class_mode = "binary",
                                                   seed = 1234, # Change to params$seed in Rmd.
                                                   target_size = target_size,
                                                   batch_size = batch_size,
                                                   color_mode = "grayscale",
                                                   subset = "validation")

# Element generate
batch <- generator_next(train.image_array_gen)

str(batch)

# Vi plotter bildene for å se hvordan data augmentation fungerer!
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  plot(as.raster(batch[[1]][i,,,]))
}

# number of training samples
train_samples <- train.image_array_gen$n
# number of validation samples
valid_samples <- val.image_array_gen$n

# Try the simple model given in cifar10_2classes (example)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", 
                padding="same",
                input_shape = image_size) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", 
                padding="same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.01)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

# Testing to compile and fit the model. 
model %>% compile(
  loss = "binary_crossentropy",
  #optimizer = "adam", # Adam fungerer dårlig her virker det som på meg!
  optimizer = optimizer_rmsprop(learning_rate = 1e-4), # decay does not seem to work very well. 
  #optimizer = optimizer_adadelta(),
  metrics = c("accuracy")
)


history <-model %>% fit( # Or fit_generator?
  train.image_array_gen,
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = val.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
)

plot(history) # Ser litt ut som overfitting! Vi har kun 668 bilder å trene på, så ikke så veldig mange!
# Prøv å: 
# - Add dropout. 
# - L2 regularization.
# - Data Augmentation (?)
# - Prøv å endre optimizer!?


# Prøver å bruke modellen fra https://shirinsplayground.netlify.app/2018/06/keras_fruits/ !
# Tok utgangspunkt i den, men har endret den litt for å prøvde å motvirke overtrening!
model3 <- keras_model_sequential() %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3,3), padding = "same", 
                activation = "relu", input_shape = image_size) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", 
                activation = "relu") %>%
  layer_batch_normalization() %>% # Vet ikke helt hva batch_normalization er?
  
  # Third hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", 
                activation = "relu", input_shape = image_size) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(256, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(units = 1, activation = "sigmoid")

summary(model3)
model3 %>% compile(
  loss = "binary_crossentropy",
  #optimizer = "adam", # Adam fungerer dårlig her virker det som på meg!
  optimizer = optimizer_rmsprop(learning_rate = 1e-4), # decay does not seem to work very well. 
  #optimizer = optimizer_adadelta(),
  metrics = c("accuracy")
)

history3 <-model3 %>% fit( 
  train.image_array_gen,
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = val.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
)

plot(history3) # Still overfitting, but not as bad as earlier!

# Prøver å bruker et pretrained nettverk også, for å se hvordan det fungerer!
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = image_size
)
conv_base

model2 <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model2)
freeze_weights(conv_base)
summary(model2)

model2 %>% compile(
  loss = "binary_crossentropy",
  #optimizer = "adam", # Adam fungerer dårlig her virker det som på meg!
  optimizer = optimizer_rmsprop(learning_rate = 1e-4), # decay does not seem to work very well. 
  metrics = c("accuracy")
)


history2 <-model2 %>% fit( # Or fit_generator?
  train.image_array_gen,
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = val.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
)


unfreeze_weights(conv_base,from ="block4_conv1")
summary(model2)

model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 2e-5),
  metrics = c("accuracy")
)

history2 <-model2 %>% fit( # Or fit_generator?
  train.image_array_gen,
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = val.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
)