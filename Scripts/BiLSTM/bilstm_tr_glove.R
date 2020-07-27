#setwd("D:/src/emotionalviolent");

#options(expressions = 5e5)
#memory.limit(size=12000000)

library(tools)
library(keras)

source(file_path_as_absolute("utils.R"))
source(file_path_as_absolute("resultshelper.R"))
source(file_path_as_absolute("datasets/gloveloader.R"))

for (i in 1:10) {

  source(file_path_as_absolute("getDados_tr.R"))

  ### Rede
  
  embedding_dims <- 100
  filters <- 164
  
  main_input <- layer_input(shape = c(maxlen), dtype = "int32")
  
  embedding_input <- 	main_input %>% 
    layer_embedding(input_dim = vocab_size, output_dim = embedding_dims, input_length = maxlen, name = "embedding")

  main_output <- embedding_input %>% 
    bidirectional(
      layer_lstm(units = 128, return_sequences = TRUE)
    ) %>%
    bidirectional(
      layer_lstm(units = 64, return_sequences = TRUE, recurrent_dropout = 0.2)
    ) %>%
    bidirectional(
      layer_lstm(units = 64)
    ) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 6, activation = 'softmax')
  
  model <- keras_model(
    inputs = c(main_input),
    outputs = main_output
  )

  embedding_dim <- 100
  embedding_matrix <- array(0, c(vocab_size, embedding_dim))
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < vocab_size) {
      embedding_vector <- embeddings_index[[word]]
      if (!is.null(embedding_vector))
        embedding_matrix[index+1,] <- embedding_vector
    }
  }

  get_layer(model, index = 1) %>%
      set_weights(list(embedding_matrix))
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    dados_train,
    to_categorical(dataTrain$categoria),
    epochs = 50,
    batch_size = 64,
    validation_split = 0.15
  )
  
  history
  
  predictions <- model %>% predict(dados_test)
  predictionsMax <- apply(predictions, 1, which.max) - 1
  matriz <- confusionMatrix(as.factor(dataTest$categoria), as.factor(predictionsMax))
  addResult(matriz)
  resultados
  dumpResults("resultados/bilstmglove_tr_50epocas.txt")
}

dumpResults("resultados/bilstmglove_tr_50epocas.txt")