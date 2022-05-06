library(png)

set.seed(1234) # Making the setting of the seed more explicit (even though it is done in the setup already).

image.path <- "./images"

# We collect the names of all the images.
normal.image.names <- list.files(paste0(image.path,"/normal"))
effusion.image.names <- list.files(paste0(image.path,"/effusion"))

img.width <- 512
img.height <- 512
# We make two tensors to fit the 500 images of both types respectively, of size 512*512 with 1 color channel. 
normal.images <- effusion.images <- array(rep(NA, 500*img.width*img.height), dim = c(500, img.width, img.height))
for (i in 1:500){
  normal.images[i,,] <- readPNG(paste0(image.path, "/normal/",normal.image.names[i]))[,,1]
  # readPNG assumes RGB images. Since ours are greyscale, R = G = B, and we can remove to color channels, which is why we have [,,1].
  # Images are too big to train the model with images in memory!! Need to reshape or go back to generators!
  # or use image_load from Keras to choose grayscale and target_size, but I do not know how to plot these e.g.
}

# We make a tensor to fit the 500 images, of size 512*512 with 1 color channel. 
for (i in 1:500){
  effusion.images[i,,] <- readPNG(paste0(image.path, "/effusion/", effusion.image.names[i]))[,,1]
  # readPNG assumes RGB images. Since ours are greyscale, R = G = B, and we can remove to color channels, which is why we have [,,1].
}

# Must be some better way than to simply load it into memory? There are iterators etc, but have not completely figured them out yet!

# We plot one normal image to check that it has worked. 
img <- normal.images[1,,]
plot(as.raster(img), main = "Normal X-ray Example", ylab = "", xlab = "")
#rasterImage(img, 1.2, 1.0, 1.8, 2.0)

# Similarly, we plot one effusion X-ray. 
img <- effusion.images[1,,]
plot(as.raster(img), main = "Effusion X-ray Example", ylab = "", xlab = "")
#rasterImage(img, 1.2, 1.0, 1.8, 2.0)

###### 
part <- 2/3
num.images <- 500
train.indices <- sample(1:num.images, size = part*num.images)
normal.train <- normal.images[train.indices,,]
normal.test <- normal.images[-train.indices,,]

effusion.train <- effusion.images[train.indices,,]
effusion.test <- effusion.images[-train.indices,,]
# Missing labels here also. 

# Used when loading all images to RAM, which should not be done!!
#gc() # Run garbage collection to remove unused variables (to restore RAM). 

x.train <- k_concatenate(list(normal.train, effusion.train), axis = 1)
# Label "normal" as 0 and "effusion" as 1.
#y.train <- to_categorical(c(rep(0, part*num.images), rep(1, part*num.images))) # for softmax and two output nodes. 
y.train <- c(rep(0, part*num.images), rep(1, part*num.images))

x.test <- k_concatenate(list(normal.test, effusion.test), axis = 1)
# Label "normal" as 0 and "effusion" as 1.
#y.test <- to_categorical(c(rep(0, num.images-round(part*num.images)), rep(1, num.images-round(part*num.images)))) # for softmax and two output nodes. 
y.test <- c(rep(0, num.images-round(part*num.images)), rep(1, num.images-round(part*num.images)))

gc() # Run Garbage Collection to free up memory from unused variables.