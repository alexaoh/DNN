

# Set wd right

wd <- "~/AEXNP/GAN"
ok_wd <- path.expand(wd)
actual_wd <- getwd()

if(actual_wd != ok_wd) setwd(path.expand(wd))

# check
getwd()

# Load GAN archicteture

source("gan_ntw.R")


# Load data

#########################
# setwd("./faces32x32")
# ok_wd <- path.expand("~/AEXNP/GAN")
# actual_wd <- getwd()

#if(actual_wd != ok_wd) setwd(path.expand("~/AEXNP/GAN"))


lf <- list.files(file.path(".", "faces32x32"))
x_train<-array(dim=c(2000,32,32,3))
for(i in 1:2000){
  x_train[i,,,]<-readImage(file.path(".", "faces32x32" ,lf[i]))
}

img<-x_train[1,,,]
imageShow(img)


# Create directory if it is not exist

results.folder <- file.path(".", "faces")
dir.create(results.folder, showWarnings = FALSE)


#########################




############################### Loads CIFAR10 data
cifar10 <- dataset_cifar10()
c(c(x_train, y_train), c(x_test, y_test)) %<-% cifar10

# Selects horses images (class 7)
class <- 7
x_train <- x_train[as.integer(y_train) == class,,,] 
# Normalizes data
x_train <- x_train / 255


# Create directory if it is not exist

results.folder <- paste0(".", "/cifar10_",class )
dir.create(results.folder, showWarnings = FALSE)


######################################################



iterations <- 500
batch_size <- 32

# Start the training loop
start <- 1


seed_latent_vectors <- matrix(rnorm(1 * latent_dim), 
                              nrow = 1, ncol = latent_dim)

for (step in 1:iterations) {
  
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  
  # Decodes them to fake images
  generated_images <- generator %>% predict(random_latent_vectors)
  
  # Combines them with real images
  stop <- start + batch_size - 1 
  real_images <- x_train[start:stop,,,]
  rows <- nrow(real_images)
  combined_images <- array(0, dim = c(rows * 2, dim(real_images)[-1]))
  combined_images[1:rows,,,] <- generated_images
  combined_images[(rows+1):(rows*2),,,] <- real_images
  
  # Assembles labels discriminating real from fake images
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))
  
  # Adds random noise to the labels -- an important trick!
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),
                                  dim = dim(labels)))
  
  # Trains the discriminator
  d_loss <- discriminator %>% train_on_batch(combined_images, labels) 
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  
  # Assembles labels that say "all real images"
  misleading_targets <- array(0, dim = c(batch_size, 1))
  
  # Trains the generator (via the gan model, where the 
  # discriminator weights are frozen)
  a_loss <- gan %>% train_on_batch( 
    random_latent_vectors, 
    misleading_targets
  )  
  
  start <- start + batch_size
  if (start > (nrow(x_train) - batch_size))
    start <- 1
  
  # Occasionally saves images
  if (step %% 5 == 0) { 
    # setwd("./models")
    
    # Create directory if it is not exist
    dir.create(file.path(results.folder, "models"), showWarnings = FALSE)
    
    # Saves model weights
    save_model_weights_hdf5(gan, paste0(results.folder, "/models/",step,"_gan.h5"))
    
    # Prints metrics
    cat("discriminator loss:", d_loss, "\n")
    cat("adversarial loss:", a_loss, "\n")  
    
    # Saves one generated image
    
    generated_images <- generator %>% predict(seed_latent_vectors)
    img<-generated_images[1,,,]
    #setwd("./generated_images")
    
    # Create directory if it is not exist
    
    dir.create(file.path(results.folder, "generated_images"), showWarnings = FALSE)
    
    writeImage(img, paste0(results.folder, "/generated_images/",   "generated_", step, ".jpg"))
    
  }
}



#
# Using the generator
#

n_images <- 16
load_model_weights_hdf5(gan,paste0(results.folder,"/models/470_gan.h5"))
latent_vectors <- matrix(rnorm(n_images * latent_dim),nrow = n_images, ncol = latent_dim)
generated_images <- generator %>% predict(latent_vectors)
imageShow(generated_images[1,,,]) 
