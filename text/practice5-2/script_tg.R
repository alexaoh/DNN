

Sys.setenv("XLA_FLAGS=--xla_gpu_cuda_data_dir"="/usr/local/cuda")
Sys.setenv("TF_XLA_FLAGS"="--tf_xla_enable_xla_devices")
Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)


library(keras)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)

maxlen <- 40


# Retrieve text
path <- get_file(
  'nietzsche.txt', 
  origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)

# Load, collapse, and tokenize text
text <- read_lines(path) %>%
  str_to_lower() %>%
  str_c(collapse = "\n") %>%
  tokenize_characters(strip_non_alphanum = FALSE, simplify = TRUE)  # mirar help

print(sprintf("corpus length: %d", length(text)))

chars <- text %>%
  unique() %>%
  sort()

print(sprintf("total chars: %d", length(chars)))  

# Cut the text in semi-redundant sequences of maxlen characters
dataset <- map(
  seq(1, length(text) - maxlen - 1, by = 3), 
  ~list(sentece = text[.x:(.x + maxlen - 1)], next_char = text[.x + maxlen])
)


dataset <- transpose(dataset)   # reconfiguracio de la llista



# Vectorization 
x <- array(0, dim = c(length(dataset$sentece), maxlen, length(chars)))
y <- array(0, dim = c(length(dataset$sentece), length(chars)))


dim(x)
dim(y)

# format one-hot
for(i in 1:length(dataset$sentece)){
  
  x[i,,] <- sapply(chars, function(x){
    as.integer(x == dataset$sentece[[i]])
  })
  
  y[i,] <- as.integer(chars == dataset$next_char[[i]])
  
}

# Exemple
x[1,,]
y[1,]


model <- keras_model_sequential()


# no embeding directe one-hot
model %>%
  layer_lstm(128, input_shape = c(maxlen, length(chars))) %>%  
  layer_dense(length(chars)) %>%
  layer_activation("softmax")     # volem predir el caracter (un dels 57)

optimizer <- optimizer_rmsprop(learning_rate = 0.01)

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)



sample_mod <- function(preds, temperature = 1){
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp(preds))
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}



#But there’s still some missing link: the keras predict function will simply give us the final 57-long 
#vector which represents the probability distribution of the 57 chars to be the next character. 
#We shall have to either choose the character of maximum value (probability) as our next character 
#or sample from this (multinomial!) distribution to get our next character. The  
#example adds a temperature (or diversity) parameter to “slide” between these two options:
#So the more we increase the temperature parameter, we make the output distribution 
#over chars more Uniform, allowing for more “diversity” in output. 
#And the more we decrease the temperature parameter, we make the output distribution 
#over chars more extreme, and we will almost surely pick the character which has the 
#maximum value for our next character.


on_epoch_end <- function(epoch, logs) {
  
  cat(sprintf("epoch: %02d ---------------\n\n", epoch))
  
  for(diversity in c(0.2, 0.5, 1, 1.2)){
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - maxlen), size = 1)
    sentence <- text[start_index:(start_index + maxlen - 1)]
    generated <- ""
    
    for(i in 1:400){
      
      x <- sapply(chars, function(x){
        as.integer(x == sentence)
      })
      x <- array_reshape(x, c(1, dim(x)))
      
      preds <- predict(model, x)
      next_index <- sample_mod(preds, diversity)
      next_char <- chars[next_index]
      
      generated <- str_c(generated, next_char, collapse = "")
      sentence <- c(sentence[-1], next_char)
      
    }
    
    cat(generated)
    cat("\n\n")
    
  }
}


print_callback <- callback_lambda(on_epoch_end = on_epoch_end)  # al final de cada epoch crida la funcio

#model<-load_model_hdf5("textgen.hdf5")

model %>% fit(
  x, y,
  batch_size = 128,
  epochs = 40,
  callbacks = print_callback
)
