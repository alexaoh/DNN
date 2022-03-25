## ---------------------------------------------------------------------------------
library(keras)
library(RColorBrewer)
library(caret)
library(readr)
library(pROC)


## ---------------------------------------------------------------------------------
setwd("/home/ajo/gitRepos/DNN/task1")
clinical <- read_delim("clinical.csv","\t", escape_double = FALSE, trim_ws = TRUE)
#gene_expression <- read_delim("gene_expression.csv","\t", escape_double = FALSE, trim_ws = TRUE)
protein_abundance <- read_delim("protein_abundance.csv","\t", escape_double = FALSE, trim_ws = TRUE)
#copy_number <- read_delim("copy_number.csv","\t", escape_double = FALSE, trim_ws = TRUE)

set1<-intersect(protein_abundance$Sample,clinical$Sample)

xclinical<-clinical[clinical$Sample%in%set1,]
xprotein<-protein_abundance[protein_abundance$Sample%in%set1,]

xclinical<-xclinical[,c(1,9)]
sel1<-which(xclinical$breast_carcinoma_estrogen_receptor_status!="Positive")
sel2<-which(xclinical$breast_carcinoma_estrogen_receptor_status!="Negative")

sel<-intersect(sel1,sel2)
xclinical<-xclinical[-sel,]

xclinical<-xclinical[-which(is.na(xclinical$breast_carcinoma_estrogen_receptor_status)),]


mprotein<-merge(xclinical,xprotein,by.x="Sample",by.y="Sample")


# pprotein
sel<-complete.cases(t(mprotein))


## ---------------------------------------------------------------------------------
set.seed(111)
training<-sample(1:nrow(mprotein),2*nrow(mprotein)/3)


xtrain<-mprotein[training,-c(1,2)]
xtest<-mprotein[-training,-c(1,2)]

xtrain<-scale(xtrain)
xtest<-scale(xtest)

ytrain<-mprotein[training,2]
ytest<-mprotein[-training,2]

ylabels<-vector()
ylabels[ytrain=="Positive"]<-1
ylabels[ytrain=="Negative"]<-0


ytestlabels<-vector()
ytestlabels[ytest=="Positive"]<-1
ytestlabels[ytest=="Negative"]<-0


## ---------------------------------------------------------------------------------
model<-keras_model_sequential() %>% 
  layer_dense(units=50,activation="relu",input_shape=c(142)) %>%
  layer_dense(units=20,activation="relu") %>%
  layer_dense(units=50,activation="relu") %>%
  layer_dense(units=142,activation="linear")


## ---------------------------------------------------------------------------------
summary(model)


## ---------------------------------------------------------------------------------
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
model %>% fit(
  x=as.matrix(xtrain),
  y=as.matrix(xtrain),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
x.hat<-predict(model,as.matrix(xtrain))


## ---------------------------------------------------------------------------------
vcor<-diag(cor(x.hat,xtrain))
hist(vcor)


## ---------------------------------------------------------------------------------
# visual inspection
plot(x.hat[,50]~xtrain[,50])


## ---------------------------------------------------------------------------------
x.hat<-predict(model,as.matrix(xtest))


## ---------------------------------------------------------------------------------
vcor<-diag(cor(x.hat,xtest))
hist(vcor)


## ---------------------------------------------------------------------------------
# visual inspection
plot(x.hat[,50]~xtest[,50])


## ---------------------------------------------------------------------------------
input_enc<-layer_input(shape = 142)
output_enc<-input_enc %>% 
  layer_dense(units=50,activation="relu") %>%
  layer_dense(units=20,activation="relu")

encoder = keras_model(input_enc, output_enc)
summary(encoder)


## ---------------------------------------------------------------------------------
input_dec = layer_input(shape = 20)
output_dec<-input_dec %>% 
  layer_dense(units=50,activation="relu") %>%
  layer_dense(units=142,activation="linear")

decoder = keras_model(input_dec, output_dec)
 
summary(decoder)


## ---------------------------------------------------------------------------------
aen_input = layer_input(shape = 142)
aen_output = aen_input %>% 
  encoder() %>% 
  decoder()
   
aen = keras_model(aen_input, aen_output)
summary(aen)


## ---------------------------------------------------------------------------------
aen %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
aen %>% fit(
  x=as.matrix(xtrain),
  y=as.matrix(xtrain),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
#Generating with Autoencoder
encoded_expression <- encoder %>% predict(as.matrix(xtrain))
decoded_expression <- decoder %>% predict(encoded_expression)


## ---------------------------------------------------------------------------------
colMain <- colorRampPalette(brewer.pal(8, "Blues"))(15)
heatmap(encoded_expression,  RowSideColors=as.character(ylabels) , col=colMain, scale="row" )


## ---------------------------------------------------------------------------------
input_enc1<-layer_input(shape = 142)
output_enc1<-input_enc1 %>% 
  layer_dense(units=50,activation="relu") 
encoder1 = keras_model(input_enc1, output_enc1)
summary(encoder1)


## ---------------------------------------------------------------------------------
input_dec1 = layer_input(shape = 50)
output_dec1<-input_dec1 %>% 
  layer_dense(units=142,activation="linear")

decoder1 = keras_model(input_dec1, output_dec1)
 
summary(decoder1)


## ---------------------------------------------------------------------------------
aen_input1 = layer_input(shape = 142)
aen_output1 = aen_input1 %>% 
  encoder1() %>% 
  decoder1()
   
sae1 = keras_model(aen_input1, aen_output1)
summary(sae1)


## ---------------------------------------------------------------------------------
sae1 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae1 %>% fit(
  x=as.matrix(xtrain),
  y=as.matrix(xtrain),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
#Generating with Autoencoder
encoded_expression1 <- encoder1 %>% predict(as.matrix(xtrain))


## ---------------------------------------------------------------------------------
input_enc2<-layer_input(shape = 50)
output_enc2<-input_enc2 %>% 
  layer_dense(units=20,activation="relu") 
encoder2 = keras_model(input_enc2, output_enc2)
summary(encoder2)


## ---------------------------------------------------------------------------------
input_dec2 = layer_input(shape = 20)
output_dec2<-input_dec2 %>% 
  layer_dense(units=50,activation="linear")

decoder2 = keras_model(input_dec2, output_dec2)
 
summary(decoder2)


## ---------------------------------------------------------------------------------
aen_input2 = layer_input(shape = 50)
aen_output2 = aen_input2 %>% 
  encoder2() %>% 
  decoder2()
   
sae2 = keras_model(aen_input2, aen_output2)
summary(sae2)


## ---------------------------------------------------------------------------------
sae2 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae2 %>% fit(
  x=as.matrix(encoded_expression1),
  y=as.matrix(encoded_expression1),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
#Generating with Autoencoder
encoded_expression2 <- encoder2 %>% predict(as.matrix(encoded_expression1))


## ---------------------------------------------------------------------------------
input_enc3<-layer_input(shape = 20)
output_enc3<-input_enc3 %>% 
  layer_dense(units=10,activation="relu") 
encoder3 = keras_model(input_enc3, output_enc3)
summary(encoder3)


## ---------------------------------------------------------------------------------
input_dec3 = layer_input(shape = 10)
output_dec3<-input_dec3 %>% 
  layer_dense(units=20,activation="linear")

decoder3 = keras_model(input_dec3, output_dec3)
 
summary(decoder3)


## ---------------------------------------------------------------------------------
aen_input3 = layer_input(shape = 20)
aen_output3 = aen_input3 %>% 
  encoder3() %>% 
  decoder3()
   
sae3 = keras_model(aen_input3, aen_output3)
summary(sae3)


## ---------------------------------------------------------------------------------
sae3 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae3 %>% fit(
  x=as.matrix(encoded_expression2),
  y=as.matrix(encoded_expression2),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
#Generating with Autoencoder
encoded_expression3 <- encoder3 %>% predict(as.matrix(encoded_expression2))


## ---------------------------------------------------------------------------------
sae_input = layer_input(shape = 142)
sae_output = sae_input %>% 
  encoder1() %>% 
  encoder2()  %>%
  encoder3() %>%
  layer_dense(5,activation = "relu")%>%
  layer_dense(1,activation = "sigmoid")
   
sae = keras_model(sae_input, sae_output)
summary(sae)


## ---------------------------------------------------------------------------------
freeze_weights(sae,from=1,to=3)


## ---------------------------------------------------------------------------------
summary(sae)


## ---------------------------------------------------------------------------------
sae %>% compile(
  optimizer = "rmsprop",
  loss = 'binary_crossentropy',
  metric = "acc"
  )


## ---------------------------------------------------------------------------------
sae %>% fit(
  x=xtrain,
  y=ylabels,
  epochs = 30,
  batch_size=64,
  validation_split = 0.2
  )


## ---------------------------------------------------------------------------------
sae %>%
  evaluate(as.matrix(xtest), ytestlabels)


## ---------------------------------------------------------------------------------
yhat <- predict(sae,as.matrix(xtest))


## ---------------------------------------------------------------------------------
yhatclass<-as.factor(ifelse(yhat<0.5,0,1))
table(yhatclass,  ytestlabels)


## ---------------------------------------------------------------------------------
confusionMatrix(yhatclass,as.factor(ytestlabels))


## ---------------------------------------------------------------------------------
roc_sae_test <- roc(response = ytestlabels, predictor =yhat)
plot(roc_sae_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae"), lty = c(1), col = c("blue"))

