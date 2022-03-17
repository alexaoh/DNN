# Autoenconder  (AE) digits from MNIST dataset

library(keras)


#### Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)

###################### Selection several digits

cifra <- c(0:9) 

x_train_cifra<-x_train[which(y_train %in% cifra),]
y_train_cifra<-y_train[which(y_train %in% cifra)]

x_test_cifra<-x_test[which(y_test %in% cifra),]
y_test_cifra<-y_test[which(y_test %in% cifra)]


dim(x_train_cifra)
length(y_train_cifra)
dim(x_test_cifra)
length(y_test_cifra)

sort(unique(y_train_cifra))
sort(unique(y_test_cifra))


######################## rescale

x_train_cifra <- x_train_cifra / 255
x_test_cifra <- x_test_cifra / 255
#y_train_cifra <- to_categorical(y_train_cifra, 10)
#y_test_cifra <- to_categorical(y_test_cifra, 10)


############################### Autoencoder

#### Encoder 

model_enc <- keras_model_sequential() 
model_enc %>%
  layer_dense(units = 128, activation = "relu", input_shape =  ncol(x_train)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") 
summary(model_enc)

#### Decoder 

model_dec <- keras_model_sequential() 
model_dec %>%
  layer_dense(units = 64, activation = "relu", input_shape =  c(32)) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = ncol(x_train), activation = "relu") 
summary(model_dec)


#### Autoencoder 

model<-keras_model_sequential()
model %>%model_enc%>%model_dec

##########################################################

summary(model)




######################################## Training 


model %>% compile(
  loss = "mean_squared_error",
  #optimizer = optimizer_rmsprop(),
  optimizer = "adam",
  metrics = c("mean_squared_error")
)

history <- model %>% fit(
  x= x_train_cifra, y = x_train_cifra,   # Autoencoder
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)

######## Prediction 

# Autoencoder
output_cifra<-predict(model,x_test_cifra)
dim(output_cifra)


# From input to encoder
enc_output_cifra<-predict(model_enc,x_test_cifra)
dim(enc_output_cifra)

# From encoder to decoder
dec_output_cifra<-predict(model_dec,enc_output_cifra)
dim(dec_output_cifra)


# Check

idx<-1
#x_test_cifra[idx,]
im<-matrix(x_test_cifra[idx,], nrow=28, ncol=28) 
image(1:28, 1:28, im, col=gray((0:255)/255))

#output_cifra[idx,]
im<-matrix(output_cifra[idx,], nrow=28, ncol=28) 
image(1:28, 1:28, im, col=gray((0:255)/255))


#dec_output_cifra[idx,]
im<-matrix(dec_output_cifra[idx,], nrow=28, ncol=28) 
image(1:28, 1:28, im, col=gray((0:255)/255))

#which.max(enc_output_cifra[idx,])
#which.min(enc_output_cifra[idx,])
#which(enc_output_cifra[idx,]==cifra)

# Encoder results
im<-matrix(enc_output_cifra[idx,], nrow=8, ncol=4) 
image(1:8, 1:4, im, col=gray((0:255)/255))


###########################################################


#dim(x_train_cifra)
#dim(x_test_cifra)

# Save encode digit in a Rdata file

save(enc_output_cifra, y_test_cifra, file=paste0("Encod_", paste(cifra, collapse = ""), ".RData"))
