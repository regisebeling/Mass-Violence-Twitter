dataTrain <- csvRead('datasets/MS_Treino_TR_Treino.csv', 20000)
dataTest <- csvRead('datasets/MS_GS_v2.csv', 20000)

allTexts <- rbind(dataTrain, dataTest)

maxlen <- 70

tokenizer <-  text_tokenizer() %>%
fit_text_tokenizer(allTexts$text)
vocab_size <- length(tokenizer$word_index) + 1
vocab_size

sequences_train <- texts_to_sequences(tokenizer, dataTrain$text)
dados_train <- pad_sequences(sequences_train, maxlen = maxlen)

sequences_test <- texts_to_sequences(tokenizer, dataTest$text)
dados_test <- pad_sequences(sequences_test, maxlen = maxlen)
word_index = tokenizer$word_index