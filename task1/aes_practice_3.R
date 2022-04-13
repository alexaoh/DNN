## ---------------------------------------------------------------------------------
library(keras)
library(RColorBrewer)
library(caret)
library(readr)
library(pROC)


## ---------------------------------------------------------------------------------
#setwd("/home/ajo/gitRepos/DNN/task1")
#clinical <- read_delim("clinical.csv","\t", escape_double = FALSE, trim_ws = TRUE)
#gene_expression <- read_delim("gene_expression.csv","\t", escape_double = FALSE, trim_ws = TRUE)
#protein_abundance <- read_delim("protein_abundance.csv","\t", escape_double = FALSE, trim_ws = TRUE)
#copy_number <- read_delim("copy_number.csv","\t", escape_double = FALSE, trim_ws = TRUE)

set1<-intersect(prot.ab$Sample,clinical$Sample)

xclinical<-clinical[clinical$Sample%in%set1,]
xprotein<-prot.ab[prot.ab$Sample%in%set1,]

xclinical<-xclinical[,c(1,9)]
colnames(xclinical) <- c("Sample", "BRCA")
sel1<-which(xclinical$BRCA!="Positive")
sel2<-which(xclinical$BRCA!="Negative")

sel<-intersect(sel1,sel2)
xclinical<-xclinical[-sel,]

xclinical<-xclinical[-which(is.na(xclinical$BRCA)),]

mprotein<-merge(xclinical,xprotein,by.x="Sample",by.y="Sample")


# pprotein
sel<-complete.cases(t(mprotein))


## ---------------------------------------------------------------------------------
set.seed(111)
training.fraction <- 0.70 # 70 % of data will be used for training. 
training<-sample(1:nrow(mprotein),training.fraction*nrow(mprotein))


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
p_input_enc1<-layer_input(shape = 142)
p_output_enc1<-p_input_enc1 %>% 
  layer_dense(units=50,activation="relu") 
prot_encoder1 = keras_model(p_input_enc1, p_output_enc1)
summary(prot_encoder1)


## ---------------------------------------------------------------------------------
p_input_dec1 <- layer_input(shape = 50)
p_output_dec1<-p_input_dec1 %>% 
  layer_dense(units=142,activation="linear")

p_decoder1 = keras_model(p_input_dec1, p_output_dec1)

summary(p_decoder1)


## ---------------------------------------------------------------------------------
p_aen_input1 <- layer_input(shape = 142)
p_aen_output1 <- p_aen_input1 %>% 
  prot_encoder1() %>% 
  p_decoder1()

sae_protab1 <- keras_model(p_aen_input1, p_aen_output1)
summary(sae_protab1)


## ---------------------------------------------------------------------------------
sae_protab1 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae_protab1 %>% fit(
  x=as.matrix(xtrain),
  y=as.matrix(xtrain),
  epochs = 100,
  batch_size=64,
  validation_split = 0.2
)


## ---------------------------------------------------------------------------------
#Generating with Autoprot_encoder
encoded_expression1 <- prot_encoder1 %>% predict(as.matrix(xtrain))


## ---------------------------------------------------------------------------------
p_input_enc2<-layer_input(shape = 50)
p_output_enc2<-p_input_enc2 %>% 
  layer_dense(units=20,activation="relu") 
prot_encoder2 = keras_model(p_input_enc2, p_output_enc2)
summary(prot_encoder2)


## ---------------------------------------------------------------------------------
p_input_dec2 <- layer_input(shape = 20)
p_output_dec2<-p_input_dec2 %>% 
  layer_dense(units=50,activation="linear")

p_decoder2 <- keras_model(p_input_dec2, p_output_dec2)

summary(p_decoder2)


## ---------------------------------------------------------------------------------
p_aen_input2 <- layer_input(shape = 50)
p_aen_output2 <- p_aen_input2 %>% 
  prot_encoder2() %>% 
  p_decoder2()

sae_protab2 <- keras_model(p_aen_input2, p_aen_output2)
summary(sae_protab2)


## ---------------------------------------------------------------------------------
sae_protab2 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae_protab2 %>% fit(
  x=as.matrix(encoded_expression1),
  y=as.matrix(encoded_expression1),
  epochs = 100,
  batch_size=64,
  validation_split = 0.2
)


## ---------------------------------------------------------------------------------
#Generating with Autoprot_encoder
encoded_expression2 <- prot_encoder2 %>% predict(as.matrix(encoded_expression1))


## ---------------------------------------------------------------------------------
p_input_enc3<-layer_input(shape = 20)
p_output_enc3<-p_input_enc3 %>% 
  layer_dense(units=10,activation="relu") 
prot_encoder3 <- keras_model(p_input_enc3, p_output_enc3)
summary(prot_encoder3)


## ---------------------------------------------------------------------------------
p_input_dec3 <- layer_input(shape = 10)
p_output_dec3<-p_input_dec3 %>% 
  layer_dense(units=20,activation="linear")

p_decoder3 <- keras_model(p_input_dec3, p_output_dec3)

summary(p_decoder3)


## ---------------------------------------------------------------------------------
p_aen_input3 <- layer_input(shape = 20)
p_aen_output3 <- p_aen_input3 %>% 
  prot_encoder3() %>% 
  p_decoder3()

sae_protab3 <- keras_model(p_aen_input3, p_aen_output3)
summary(sae_protab3)


## ---------------------------------------------------------------------------------
sae_protab3 %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)


## ---------------------------------------------------------------------------------
sae_protab3 %>% fit(
  x=as.matrix(encoded_expression2),
  y=as.matrix(encoded_expression2),
  epochs = 100,
  batch_size=64,
  validation_split = 0.2
)


## ---------------------------------------------------------------------------------
#Generating with Autoprot_encoder
encoded_expression3 <- prot_encoder3 %>% predict(as.matrix(encoded_expression2))


## ---------------------------------------------------------------------------------
sae_protab_input = layer_input(shape = 142, name = "prot.mod")
sae_protab_output = sae_protab_input %>% 
  prot_encoder1() %>% 
  prot_encoder2()  %>%
  prot_encoder3() %>%
  layer_dense(5,activation = "relu") %>%
  layer_dense(1,activation = "sigmoid")

sae_protab = keras_model(sae_protab_input, sae_protab_output)
summary(sae_protab)


## ---------------------------------------------------------------------------------
freeze_weights(sae_protab,from=1,to=4)
# Freeze weights in order to only finetune weights to two fully connected dense layers (of 5 nodes + output layer).


## ---------------------------------------------------------------------------------
summary(sae_protab)


## ---------------------------------------------------------------------------------
sae_protab %>% compile(
  optimizer = "rmsprop",
  loss = 'binary_crossentropy',
  metric = "acc"
)


## ---------------------------------------------------------------------------------
sae_protab %>% fit(
  x=xtrain,
  y=ylabels,
  epochs = 50,
  batch_size=64,
  validation_split = 0.2
)


## ---------------------------------------------------------------------------------
sae_protab %>%
  evaluate(as.matrix(xtest), ytestlabels)


## ---------------------------------------------------------------------------------
yhat <- predict(sae_protab,as.matrix(xtest))


## ---------------------------------------------------------------------------------
yhatclass<-as.factor(ifelse(yhat<0.5,0,1))
table(yhatclass,  ytestlabels)


## ---------------------------------------------------------------------------------
confusionMatrix(yhatclass,as.factor(ytestlabels))


## ---------------------------------------------------------------------------------
roc_sae_protab_test <- roc(response = ytestlabels, predictor =yhat)
plot(roc_sae_protab_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae_protab"), lty = c(1), col = c("blue"))
