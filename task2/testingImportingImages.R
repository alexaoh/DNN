library(tfdatasets)
library(keras)

# We load the files as a TensorFlow Dataset, following the guide
# https://tensorflow.rstudio.com/tutorials/beginners/load/load_image/
# This seems like a way better way of doing it, because it uses an iterator to only load one and one image!

normal.list.ds <- file_list_dataset("./normal/*")
effusion.list.ds <- file_list_dataset("./effusion/*")

decode_img <- function(file_path, height = 512, width = 512) {
  
  size <- as.integer(c(height, width))
  
  file_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$convert_image_dtype(dtype = tf$float64) %>% 
    tf$image$resize(size = size)
}

normal.preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    to_categorical(TRUE)
  )
}

effusion.preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    to_categorical(FALSE) # This does not make sense the way I have done it here, since it is all FALSE. 
  )
}

normal.labeled.ds <- normal.list.ds %>% 
  dataset_map(normal.preprocess_path, num_parallel_calls = tf$data$experimental$AUTOTUNE)

effusion.labeled.ds <- effusion.list.ds %>% 
  dataset_map(effusion.preprocess_path, num_parallel_calls = tf$data$experimental$AUTOTUNE)

check.normal <- normal.labeled.ds %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() 

check.normal
plot(as.raster(as.array(check.normal[[1]])))


check.effusion <- effusion.labeled.ds %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() 

check.effusion
plot(as.raster(as.array(check.effusion[[1]])))


# Prøver å bygge en modell med dette jeg har gjort ifølge guiden ovenfor:
model <- keras_model_sequential() %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax")

model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )


# Does not work to fit with this "iterator data". Not sure how to solve the problem I am having with importing the data tbh!
model %>% 
  fit(
    normal.labeled.ds %>% dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE),
    epochs = 5,
    verbose = 2
  )


##########################################################################################################################
# Prøver med generatorer:
testing <- image_dataset_from_directory("images", label_mode = "binary")
names(testing)
testing$class_names
# Vet ikke helt hvordan jeg kan bruke denne datatype som jeg har ovenfor?

# Hva med funksjonene nedenfor?
# image_data_generator sammen med flow_images_from_directory

# Da må funksjonen: fit_generator brukes når modellen skal fittes!

# https://shirinsplayground.netlify.app/2018/06/keras_fruits/

################ Prøver å følge guiden ovenfor delvis!

# image size to scale down to (original images are 512 x 512 px)
img_width <- 256
img_height <- 256
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# optional data augmentation
train_data_gen <- image_data_generator(
  rescale = 1/255, 
  validation_split = 1/3,
  #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fill_mode = "nearest"
)

# Then we load the data using the generators. 
# training images.
train.image_array_gen <- flow_images_from_directory("images", 
                                                    train_data_gen,
                                                    class_mode = "binary",
                                                    seed = params$seed, 
                                                    target_size = target_size,
                                                    subset = "training")

# testing (or possibly validation) images. 
test.image_array_gen <- flow_images_from_directory("images", 
                                                   train_data_gen,
                                                   class_mode = "binary",
                                                   seed = params$seed, 
                                                   target_size = target_size,
                                                   subset = "validation")
names(train.image_array_gen)
table(factor(train.image_array_gen$classes))
train.image_array_gen$class_indices
train.image_array_gen$subset
train.image_array_gen$n

names(test.image_array_gen)
table(factor(test.image_array_gen$classes))
test.image_array_gen$class_indices
test.image_array_gen$subset
test.image_array_gen$n

# number of training samples
train_samples <- train.image_array_gen$n
# number of validation samples
valid_samples <- test.image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# Tester å bygge en modell med dette, tar dritlang tid pga definisjonen av modellen!
model <- keras_model_sequential() %>%
  layer_conv_2d(filters=32, kernel_size=3, activation = "relu", input_shape=c(img_width, img_height, channels)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
summary(model)

# compile
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Den tar dritlang tid å trene haha!
hist <- model %>% fit_generator( # Jeg får beskjed om å bruke fit i stedet for fit_generator, fordi sistnevnte er deprecated!
  # training data
  train.image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = test.image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)


# Veldig dårlig modell! Og jeg har ikke noe test-data å jobbe med?!
plot(hist)
