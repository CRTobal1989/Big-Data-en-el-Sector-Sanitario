optimizacion = function(Mtrain, Ytrain, Mval, Yval, Nhidden, decay, repetitions) {
  
  # Matriz de resultados
  matriz_op <- matrix(numeric(length(decay) * length(Nhidden)), nrow = length(decay))
  rownames(matriz_op) <- paste("decay =", decay)
  colnames(matriz_op) <- paste("Num_capas =", Nhidden)
  
  for (i in 1:length(decay)) {
    for (j in 1:length(Nhidden)) {
      Met_entr <- numeric(repetitions)
      for (k in 1:repetitions) {
        
        # Entrenamiento con nnet()
        modelo <- nnet(
          Mtrain, Ytrain,     # Matrices de dato de entrenamiento
          size = Nhidden[j],  # Número de neuronas en la capa oculta
          decay = decay[i],   # Parámetro de regularización
          maxit = 500,        # Iteraciones máximas
          trace = FALSE,      # Desactiva los mensajes en consola
          MaxNWts = 10000     # Número máximo de pesos
        )
        
        # Predicción sobre los datos de validación
        pred <- predict(modelo, Mval)
        pred = round(pred)
        
        # Matriz de confusión
        matriz_confusion <- table(Real = Yval, Predicho = pred)
        
        # Calcular accuracy
        accuracy <- sum(diag(matriz_confusion)) / sum(matriz_confusion)
        
        # Almacenar la métrica de rendimiento
        Met_entr[k] <- accuracy
      }
      
      # Asignar promedio de rendimiento a la matriz
      matriz_op[i, j] <- mean(Met_entr)
    }
  }
  
  return(matriz_op)
}