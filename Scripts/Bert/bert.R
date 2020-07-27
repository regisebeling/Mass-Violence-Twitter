library(tools)
library(keras)
source(file_path_as_absolute("utils.R"))
source(file_path_as_absolute("resultshelper.R"))

library(zeallot)
library(dplyr)

Sys.setenv(TF_KERAS=1) 
# make sure we use python 3
reticulate::use_python('/usr/bin/python3', required=T)
# to see python version
reticulate::py_config()


pretrained_path = '/var/www/html/uncased_L-12_H-768_A-12'
config_path = file.path(pretrained_path, 'bert_config.json')
checkpoint_path = file.path(pretrained_path, 'bert_model.ckpt')
vocab_path = file.path(pretrained_path, 'vocab.txt')

seq_length = 70L
bch_size = 64
#epochs = 10
epochs = 4
learning_rate = 1e-4

for (i in 1:10) {
  if (i <= 8) {
    next;
  }
  DATA_COLUMN = 'text'
  LABEL_COLUMN = 'categoria'

  train = data.table::fread('datasets/MS_Treino.csv')
  test = data.table::fread('datasets/MS_GS_v2.csv')
  ### Rede

  library(reticulate)
  k_bert = import('keras_bert')
  token_dict = k_bert$load_vocabulary(vocab_path)
  tokenizer = k_bert$Tokenizer(token_dict)

  model = k_bert$load_trained_model_from_checkpoint(
  config_path,
  checkpoint_path,
  training=T,
  trainable=T,
  seq_len=seq_length)

  tokenize_fun = function(dataset) {
    c(indices, target, segments) %<-% list(list(),list(),list())
    for ( i in 1:nrow(dataset)) {
      c(indices_tok, segments_tok) %<-% tokenizer$encode(dataset[[DATA_COLUMN]][i], max_len=seq_length)
      indices = indices %>% append(list(as.matrix(indices_tok)))
      target = target %>% append(dataset[[LABEL_COLUMN]][i])
      segments = segments %>% append(list(as.matrix(segments_tok)))
    }
    return(list(indices,segments, target))
  }

  dt_data = function(data){
    c(x_train, x_segment, y_train) %<-% tokenize_fun(data)
    return(list(x_train, x_segment, y_train))
  }

  c(x_train,x_segment, y_train) %<-% dt_data(train)

  train = do.call(cbind,x_train) %>% t()
  segments = do.call(cbind,x_segment) %>% t()
  targets = do.call(cbind,y_train) %>% t()
  concat = c(list(train ),list(segments))

  c(x_test,x_segment_test, y_test) %<-% dt_data(test)

  test_validate = do.call(cbind, x_test) %>% t()
  segments_test = do.call(cbind, x_segment_test) %>% t()
  targets_test = do.call(cbind, y_test) %>% t()

  concat_test = c(list(test_validate ),list(segments_test))

  c(decay_steps, warmup_steps) %<-% k_bert$calc_train_steps(
    targets %>% length(),
    batch_size=bch_size,
    epochs=epochs
  )

  library(keras)

  input_1 = get_layer(model,name = 'Input-Token')$input
  input_2 = get_layer(model,name = 'Input-Segment')$input
  inputs = list(input_1,input_2)

  dense = get_layer(model,name = 'NSP-Dense')$output

  outputs = dense %>% layer_dense(units=6L, activation='softmax',
                                  kernel_initializer=initializer_truncated_normal(stddev = 0.02),
                                  name = 'output')

  model = keras_model(inputs = inputs,outputs = outputs)

  model %>% compile(
    k_bert$AdamWarmup(decay_steps=decay_steps, 
                      warmup_steps=warmup_steps, lr=learning_rate),
    loss = 'categorical_crossentropy',
    metrics = 'accuracy'
  )

  history <- model %>% fit(
    concat,
    to_categorical(targets),
    epochs=epochs,
    batch_size=bch_size, validation_split=0.15)
  
  history
  
  predictions <- model %>% predict(concat_test)
  predictionsMax <- apply(predictions, 1, which.max) - 1
  matriz <- confusionMatrix(factor(targets_test, levels = c("0", "1", "2", "3", "4", "5")), factor(predictionsMax, levels = c("0", "1", "2", "3", "4", "5")))
  addResult(matriz)
  resultados
  dumpResults("bert.txt")
}

dumpResults("bert.txt")
