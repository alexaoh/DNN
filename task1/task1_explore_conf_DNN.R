# Task 1 - Point 5
# Explore if 5, 10 or 20 nodes in the fully connected layer gives the best results in the classifier. 

library(keras)
library(readr)
library(pROC)

# Load and clean data before defining the model. The code is identical to in the main Rmd file. 
## ---------------------------------------------------------------------------------
# gene.exp <- read_delim("gene_expression.csv", "\t", escape_double = FALSE, trim_ws = TRUE)
# prot.ab <- read_delim("protein_abundance.csv", "\t", escape_double = FALSE, trim_ws = TRUE)
# clinical <- read_delim("clinical.csv","\t", escape_double = FALSE, trim_ws = TRUE)
# 
# gene.exp2 <- gene.exp[complete.cases(gene.exp),]
# 
# ## ---------------------------------------------------------------------------------
# int4 <- intersect(gene.exp2$Sample, clinical$Sample)
# 
# ## ---------------------------------------------------------------------------------
# chosen.data <- int4 # Change this definition after deciding which data set to use!
# xclin <- clinical[,c(1,9)]
# colnames(xclin) <- c("Sample", "BRCA")
# xclin <- xclin[clinical$Sample %in% chosen.data, ] 
# #xprot <- prot.ab[prot.ab$Sample%in%chosen.data,]
# xgene <- gene.exp2[gene.exp2$Sample %in% chosen.data, ]
# 
# sel1 <- which(xclin$BRCA != "Positive")
# sel2 <- which(xclin$BRCA != "Negative")
# sel <- intersect(sel1,sel2) # Find values of BRCA that are not negative or positive. 
# # In this case these values are either "Indeterminate" or "Not Performed".
# xclin <- xclin[-sel,] # Remove the rows with non-valid data for BRCA. 
# xclin <- xclin[-which(is.na(xclin$BRCA)),] # Also remove rows with missing data for BRCA. 
# 
# # Join the (cleaned) clinical data and the gene expression data on "Sample".
# mgene <- merge(xclin, xgene, by.x = "Sample", by.y = "Sample")
# 
# ## ---------------------------------------------------------------------------------
# percentage <- round(dim(mgene[,-c(1,2)])[[2]]*0.25) # Find how many variables correspond to 25%. 
# variances <- apply(X=mgene[,-c(1,2)], MARGIN=2, FUN=var) # Find empirical variance in each of the variables (genes).
# sorted <- sort(variances, decreasing=TRUE, index.return=TRUE)$ix[1:percentage] # Sort from highest to lowest variance and select the top 25% indices. 
# mgene.lvar <- mgene[, c(1,2,sorted)] # Select the 25% largest variance variables using the indices found above. 
# 
# ## ---------------------------------------------------------------------------------
# set.seed(111)
# training.fraction <- 0.7 # 70 % of data will be used for training. 
# training <- sample(1:nrow(mgene.lvar),nrow(mgene.lvar)*training.fraction) 
# 
# xtrain <- mgene.lvar[training,-c(1,2)]
# xtest <- mgene.lvar[-training,-c(1,2)]
# 
# # Scaling for better numerical stability. 
# # This is a standard "subtract mean and divide by standard deviation" scaling. 
# xtrain <- scale(data.matrix(xtrain)) 
# xtest <- scale(data.matrix(xtest))
# 
# # Pick out labels for train and test set. 
# ytrain <- mgene.lvar[training,2]
# ytest <- mgene.lvar[-training,2]
# 
# # Change labels to numerical values in train and test set. 
# ylabels <- c()
# ylabels[ytrain=="Positive"] <- 1
# ylabels[ytrain=="Negative"] <- 0
# 
# ytestlabels <- c()
# ytestlabels[ytest=="Positive"] <- 1
# ytestlabels[ytest=="Negative"] <- 0

data.train <- read.csv("train_for5.csv")[,-1]
data.test <- read.csv("test_for5.csv")[,-1]
xtrain1 <- data.matrix(data.train[,-1])
ylabels <- data.matrix(data.train[,1])
xtest <- data.matrix(data.test[,-1])
ytestlabels <- data.matrix(data.test[,1])
percentage <- dim(xtrain)[[2]]

############## Hyperparameter Flags
FLAGS <- flags(
  flag_integer("units",5) # Set default to 5 units in the fully connected layer. 
)

# Recreate the pretrained autoencoder from the Rmd file. 
# We define the same encoders. 

input_enc1 <- layer_input(shape = percentage)
output_enc1 <- input_enc1 %>% 
  layer_dense(units=1000,activation="relu") 
encoder1 <- keras_model(input_enc1, output_enc1)

input_enc2 <- layer_input(shape = 1000)
output_enc2 <- input_enc2 %>% 
  layer_dense(units=100,activation="relu") 
encoder2 <- keras_model(input_enc2, output_enc2)

input_enc3 <- layer_input(shape = 100)
output_enc3 <- input_enc3 %>% 
  layer_dense(units=50,activation="relu") 
encoder3 <- keras_model(input_enc3, output_enc3)

# Build the same autoencoder architecture as earlier. 
sae_input1 <- layer_input(shape = percentage)
sae_output1 <- sae_input1 %>% 
  encoder1() %>% 
  encoder2() %>%
  encoder3()

new.model <- keras_model(sae_input1, sae_output1)

# Load the weights that were saved after training in Rmd. 
load_model_weights_hdf5(new.model, "sae.hdf5")

# EXTRA: Make sure that the predictions from the model in Rmd and this model are the same. Good!
new_predictions <- predict(new.model, xtest)
yhat <- read.csv("predictions.csv")
yhat <- yhat[,-1]
colnames(yhat) <- c()
all.equal(data.matrix(yhat), new_predictions, check.attributes = F)

# Add the fully connected layers to the end of the autoencoder. 
sae_output2 <- sae_output1 %>%
   layer_dense(FLAGS$units,activation = "relu") %>% # Couple with fully connected layers (DNN).
   layer_dense(1,activation = "sigmoid")

sae <- keras_model(sae_input1, sae_output2)
summary(sae)

freeze_weights(sae,from=1,to=4) # Freeze the weights (pre-training using the SAE).
summary(sae)

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
  evaluate(xtest, ytestlabels)


## ---------------------------------------------------------------------------------
yhat <- predict(sae,xtest)
yhatclass<-as.factor(ifelse(yhat<0.5,0,1))
confusionMatrix(yhatclass,as.factor(c(ytestlabels)))


## ---------------------------------------------------------------------------------
roc_sae_test <- roc(response = as.numeric(ytestlabels), predictor = as.numeric(yhat))
plot(roc_sae_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae"), lty = c(1), col = c("blue"))
