#time-series-data-prediction
library("forecast")
library("Matrix")
library("caret")

#set the working folder
setwd("/Users/aravind/MS/DM/R")

#get the training data from "product_distribution_training_set.txt" file
fileData<-as.matrix(read.table("product_distribution_training_set.txt",header=FALSE))

#transpose the imported training data file
transposeData<-t(fileData)

#Creating training data matrix 
trainingData <- matrix(data=NA, nrow= nrow(transposeData)-1, ncol=ncol(transposeData)+1)

# calculating sum of all 100 product id and store the same in 1st collumn of training data matrix
for(i in c(1:nrow(transposeData)-1)){
  trainingData[,1][i] <- sum(fileData[,i+1])
}

# store the 100 product id data in rest of the training data matrix
for(i in c(1:ncol(transposeData))){
  trainingData[,i+1] <- transposeData[,i][2:nrow(transposeData)]
}

#Creating result data matrix
resultData <- matrix(data=NA, nrow= 30, ncol=101)

#put zero to i,i row & collumn of the result matrix
resultData[1,1] <- 0

#To store the product id's in 1st column
resultData[1,-(1:1)] <- transposeData[1,]

# Now to perform time series prediction for all 101 collumns in training data matrix(incuding overall prediction and individual prediction)
for(i in 1:101){
  
  #time serice data formation for each collumn wise
  ts <- ts(ts(trainingData[,i], frequency = 1),frequency = 1)
  
  #To form fourier regression model for smoothening of the data for model training
  reg0 <- fourier(ts(ts, frequency = 90), K=4)
  
  #To form fourier regression model for smoothening of the data for model forcast
  reg1 <- fourier(ts(ts, frequency = 365), K=4, h=29)
  
  #Neural Network data training
  nnetarFit <- nnetar(ts, xreg=reg0, size = 20, repeats = 50)
  
  #Neural Network data forcasting
  forecastNnetar <- forecast(nnetarFit, xreg = reg1, h=29)
  
  #Neural Network accuracy calculation
  accNnetar <- accuracy(forecastNnetar)
  
  #plot(forecastNnetar)
  
  #ARIMA data training
  autoArimaFit <- auto.arima(ts,  xreg=reg0)
  
  #ARIMA data forcasting
  forecastArima <- forecast(autoArimaFit, xreg = reg1, h=29)
  
  #ARIMA accuracy calculation
  accArima <- accuracy(forecastArima)
  
  #plot(forecastArima)
  
  # To select correct model based on accuracy 
  if(accNnetar[1,2] <= accArima[1,2]) {
    forecastResult <- as.numeric(forecastNnetar$mean)
  } else {
    forecastResult <- as.numeric(forecastArima$mean)
  }
  
  #Removing negative values from forecast result
  forecastResult[forecastResult<0]<-0
  
  #Rounding the forecast result
  forecastResult<-round(forecastResult)
  
  #Converting the forcast result to matrix
  forecastMatrix = as.matrix(forecastResult)

  #Storing the forcast matrix in result matrix collumn wise
  resultData[-(1:1),i] <-  forecastMatrix[,1]
  print(i)
}

#Transposing resultData matrix to get back to the intial form
finalData<-t(resultData)

#Forming output
output<-finalData

#writting output to working folder 
write.table(output,file="output.txt",sep = "\t",quote = F,row.names = F,col.names = F)
