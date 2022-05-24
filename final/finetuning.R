# Final Project - Hyperparameter Tuning. 

############## Hyperparameter Flags
FLAGS <- flags(
  flag_numeric("learning_rate", 0.001), # Set default lr. 
  flag_numeric("dropout_rate", 0.2), # Set default dropout rate. 
  flag_integer("n_dense", 1024) # Set default dropout rate. 
)

# Load the data and build the model, exactly the same as in "model.R"
global.seed <- 2022 # Use this as the seed everywhere. 
set.seed(global.seed)
path.train <- "train/"
path.test <- "test/"
label.list <- dir(path.train)
output.n <- length(label.list) # as we can see, we have 50 different birds in train images.
#length(dir(path.test)) # The same is the case for the testing images, as seen below.
#all(label.list == dir(path.test))
#save(label.list, file="label_list.RData") # Save the list of names on disk for later.

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
model <- model.function(learning_rate = FLAGS$learning_rate, 
                        dropoutrate = FLAGS$dropout_rate, 
                        n_dense = FLAGS$n_dense)
#summary(model)

# We train the model.
hist <- model %>% fit(
  train.images,
  steps_per_epoch = train.images$n %/% batch_size,# Integer division.
  epochs = epochs,
  validation_data = validation.images,
  validation_steps = validation.images$n %/% batch_size # Integer division.
)


cat("The learning rate was ", FLAGS$learning_rate)
cat("The dropout rate was ", FLAGS$dropput_rate)
cat("The n_dense was ", FLAGS$n_dense)
