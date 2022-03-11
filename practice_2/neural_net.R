library(keras)

# Set hyperparameter flags.
FLAGS <- flags(
  flag_integer("model", 1)
)

data <- read.csv("../practice_1/parkinsons_updrs.data")
data$severity <- data$total_UPDRS > 25
summary(data$severity)
biom_voice_mesures <- names(data)[7:22]
mydata <- data[, biom_voice_mesures]

normalize <- function(x) {
  return((x- min(x))/(max(x)-min(x)))
}

mydata <- as.data.frame(lapply(mydata, normalize))
ratio <- 0.7
sample.size <- floor(nrow(data) * ratio)
train.indices <- sample(1:nrow(data), size = sample.size)
train <- mydata[train.indices, ]
test <- mydata[-train.indices, ]
x_train <- data.matrix(train)
y_train <- as.numeric(data[train.indices,]$severity)
x_test <- data.matrix(test)
y_test <- as.numeric(data[-train.indices,]$severity)

set.seed(1)

# defining the model and layers.
if (FLAGS$model == 1){
  model <- keras_model_sequential() %>%
    layer_dense(units = 10, activation = 'relu', input_shape = c(ncol(x_train))) %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
} else if (FLAGS$model == 2){
  model <- keras_model_sequential() %>%
    layer_dense(units = 20, activation = 'relu', input_shape = c(ncol(x_train))) %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dense(units = 5, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
} else if (FLAGS$model == 3){
  # Model 1 with dropout regularization between the layers. 
  model <- keras_model_sequential() %>%
    layer_dense(units = 10, activation = 'relu', input_shape = c(ncol(x_train))) %>%
    layer_dropout (rate = 0.4) %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dropout (rate = 0.4) %>%
    layer_dense(units = 1, activation = 'sigmoid')
} else if (FLAGS$model == 4){
  # Model 2 with dropout regularization between the layers. 
  model <- keras_model_sequential() %>%
    layer_dense(units = 20, activation = 'relu', input_shape = c(ncol(x_train))) %>%
    layer_dropout (rate = 0.4) %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dropout (rate = 0.4) %>%
    layer_dense(units = 5, activation = 'relu') %>%
    layer_dropout (rate = 0.4) %>%
    layer_dense(units = 1, activation = 'sigmoid')
} else if (FLAGS$model == 5){
  # Model 1 with l2 regularization on the hidden layers. 
  model <- keras_model_sequential() %>%
    layer_dense(units = 10, activation = 'relu', input_shape = c(ncol(x_train)), kernel_regularizer = regularizer_l2(l = 0.01)) %>%
    layer_dense(units = 10, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0.01)) %>%
    layer_dense(units = 1, activation = 'sigmoid')
} else if (FLAGS$model == 6){
  # Model 2 with l2 regularization on the hidden layers. 
  model <- keras_model_sequential() %>%
    layer_dense(units = 20, activation = 'relu', input_shape = c(ncol(x_train)), kernel_regularizer = regularizer_l2(l = 0.01)) %>%
    layer_dense(units = 10, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0.01)) %>%
    layer_dense(units = 5, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0.01)) %>%
    layer_dense(units = 1, activation = 'sigmoid')
}

summary(model)

# compile (define loss and optimizer)
model %>% compile(loss = 'binary_crossentropy',
                  optimizer = optimizer_rmsprop(),
                  metrics = c('accuracy'))

# train (fit)
history <- model %>% fit(x_train, y_train, epochs = 100, 
                         batch_size = 256, validation_split = 0.2)

# plot
plot(history)

# evaluate on training data.
score.train <- model %>% evaluate(x_train, y_train, verbose = 0)

cat('Train loss:', score.train[1], '\n')
cat('Train accuracy:', score.train[2], '\n')

# evaluate on testing data. 
score.test <- model %>% evaluate(x_test, y_test, verbose = 0)

cat('Test loss:', score.test[1], '\n')
cat('Test accuracy:', score.test[2], '\n')
