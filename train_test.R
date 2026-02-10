train_test = function(Mtrain,Ytrain,Mtest,Ytest,Nhidden_opt,decay_opt)
{
  
  # Entrenamiento con nnet()
  modelo <- nnet(
    Mtrain, Ytrain,     # Matrices de dato de entrenamiento
    size = Nhidden_opt,  # Número de neuronas que máximiza el accuracy
    decay = decay_opt,   # Parámetro de regularización
    maxit = 500,        # Iteraciones máximas
    trace = FALSE,      # Desactiva los mensajes en consola
    MaxNWts = 10000     # Número máximo de pesos
  )
  
  # Predicción sobre los datos de test
  pred <- predict(modelo, Mtest)
  pred = round(pred)
  
  # Matriz de confusión
  matriz_confusion <- table(Real = Ytest, Predicho = pred)
  
  return(matriz_confusion)
}