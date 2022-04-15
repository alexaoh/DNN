# Training an image recognizer on MNIST data
# CNN architecture

# Install & libraries
#devtools::install_github("rstudio/keras")
library(keras)
#install_keras()

#install.packages("caret")
library(caret)


# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 15

# Input image dimensions
img_rows <- 28
img_cols <- 28


# input layer: use MNIST images
mnist <- dataset_mnist()
x_train <- mnist$train$x; y_train <- mnist$train$y
x_test <- mnist$test$x; y_test <- mnist$test$y

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

################# Define Model ##################################

# defining the model and layers
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                activation = 'relu', input_shape = input_shape) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

summary(model)

# compile (define loss and optimizer)
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
# train (fit)
history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

plot(history)

# evaluate
scores <- model %>% evaluate(x_test, y_test, verbose=0)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

#predict
# keras/tensorflow version < 2.6
#y_pred <- model %>% predict_classes(x_test)

# keras/tensorflow version >= 2.6
# se obtiene un objeto tf.tensor
y_pred <- model %>% predict(x_test) %>% k_argmax()
# se pasa a vector
# https://tensorflow.rstudio.com/guide/tensorflow/tensors/
y_pred <- y_pred %>% shape() %>% unlist()

y_pred[1:100]


#confusion Matrix

confusionMatrix(as.factor(mnist$test$y), as.factor(y_pred))
