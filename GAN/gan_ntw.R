library(keras)
library(OpenImageR)


latent_dim <- 100
height <- 32
width <- 32
channels <- 3

generator_input <- layer_input(shape = c(latent_dim))

generator_output <- generator_input %>% 
  
  # First, transform the input into a 16x16 128-channels feature map
  layer_dense(units = 128 * 8 * 8) %>%
  layer_activation_leaky_relu(alpha = 0.2) %>% 
  layer_reshape(target_shape = c(8, 8, 128)) %>% 
  
  # Then, add a convolution layer
  layer_conv_2d_transpose(filters = 128, kernel_size = 4, strides = 2,
                padding = "same") %>% 
  layer_activation_leaky_relu(alpha = 0.2) %>% 
  
  # Upsample to 32x32
  layer_conv_2d_transpose(filters = 256, kernel_size = 4, 
                          strides = 2, padding = "same") %>% 
  layer_activation_leaky_relu(alpha = 0.2) %>% 
  
  # Produce a 32x32 1-channel feature map
  layer_conv_2d(filters = channels, kernel_size = 7,
                activation = "tanh", padding = "same")

generator <- keras_model(generator_input, generator_output)
summary(generator)



example_latent_vectors <- matrix(rnorm(1 * latent_dim), 
                              nrow = 1, ncol = latent_dim)
generated_images <- generator %>% predict(example_latent_vectors)
img<-generated_images[1,,,]
imageShow(img)


# 

discriminator_input <- layer_input(shape = c(height, width, channels))

discriminator_output <- discriminator_input %>% 
  layer_conv_2d(filters = 64, kernel_size = 3,strides = 2,padding='same') %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_dropout(rate = 0.4) %>%
  
  layer_conv_2d(filters = 64, kernel_size = 3, strides = 2,padding='same') %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_dropout(rate = 0.4) %>%
  
  layer_flatten() %>%
 
  # Classification layer
  layer_dense(units = 1, activation = "tanh")

discriminator <- keras_model(discriminator_input, discriminator_output)
summary(discriminator)

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer <- optimizer_rmsprop( 
  learning_rate = 0.0008, 
  clipvalue = 1.0,
  decay = 1e-8
)

discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)


# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
freeze_weights(discriminator) 

gan_input <- layer_input(shape = c(latent_dim))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)

summary(gan)

gan_optimizer <- optimizer_rmsprop(
  learning_rate = 0.0004, 
  clipvalue = 1.0, 
  decay = 1e-8
)

gan %>% compile(
  optimizer = gan_optimizer, 
  loss = "binary_crossentropy"
)


