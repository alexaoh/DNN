library(keras)
library(mixOmics)


data(nutrimouse)
matrix1<-as.matrix(nutrimouse$gene)
matrix2<-as.matrix(nutrimouse$lipid)

train1<-array(dim=c(40,120,1))
train1[,,1]<-matrix1

train2<-array(dim=c(40,21,1))
train2[,,1]<-matrix2

ytrain<-as.array(as.numeric(nutrimouse$genotype)-1)
table(ytrain)
ytrain<-to_categorical(ytrain,2)

space_dim_1 <- 120
model_input_1 <-layer_input(shape = c(space_dim_1),name="omics1")
model_output_1 <- model_input_1 %>% 
  layer_dense(units=50, activation = "relu")  

space_dim_2 <- 21
model_input_2 <-layer_input(shape = c(space_dim_2),name="omics2")
model_output_2 <- model_input_2 %>% 
  layer_dense(units=5, activation = "relu")


concatenated<-layer_concatenate(list(model_output_1,model_output_2))
model_output<-concatenated %>% 
#  layer_dense(10,"relu") %>% 
  layer_dense(units=2,activation = "softmax")

model<-keras_model(list(model_input_1,model_input_2), model_output)
summary(model)


# 

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "acc"
)


# training 

model %>% fit(
  list(omics1=train1,omics2=train2), ytrain,
  epochs = 75, batch_size = 16, validation_split = 0.2
)



# training_on_batch

for(i in 1:100){
  
  #sel1<-sample(1:40,16)
  sel1<-sample(1:40,32,replace=T)
  batchdata1<-train1[sel1,,1]
  batchdata1<-array_reshape(batchdata1,dim=c(32,120,1))
  batchdata2<-train2[sel1,,1]
  batchdata2<-array_reshape(batchdata2,dim=c(32,21,1))
  combined<-list(omics1=batchdata1,omics2=batchdata2)
  batch.y<-ytrain[c(sel1),]
  
  
  dloss<-model %>% train_on_batch(combined, batch.y)
  print(dloss)
  
  
}
