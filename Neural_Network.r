######################################## problem1 ####################################################
##### Neural Networks 
library(readr)
# Load the dataset
startups_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Neural Network\\50_Startups.csv")
colnames(startups_data)
#EDA
#delete state column
startups_data <- startups_data[-4]

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
startups_norm <- as.data.frame(lapply(startups_data, normalize))

# create training and test data
startups_train <- startups_norm[1:35, ]
startups_test <- startups_norm[36:50, ]

## Training a model on the data ----
# train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# simple ANN with only a single hidden neuron
startups_model <- neuralnet(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,data = startups_train)


# visualize the network topology
plot(startups_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(startups_model, startups_test[1:3])
# obtain predicted Profit values
str(results_model)
predicted_Profit <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_Profit, startups_test$Profit)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
startups_model2 <- neuralnet(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,data = startups_train, hidden = 5)


# plot the network
plot(startups_model2)

# evaluate the results as we did before
model_results2 <- compute(startups_model2, startups_test[1:3])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, startups_test$Profit)

############################### problem2 #####################################
##### Neural Networks 
library(readr)
# Load the dataset
fireforest_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Neural Network\\fireforests.csv")
colnames(fireforest_data)
#EDA

#Label Encoding for month and day
factors <- factor(fireforest_data$month)
fireforest_data$month <- as.numeric(factors)

factors <- factor(fireforest_data$day)
fireforest_data$day <- as.numeric(factors)

str(fireforest_data)
# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
fireforest_norm <- as.data.frame(lapply(fireforest_data, normalize))

# create training and test data
fireforest_train <- fireforest_norm[1:414, ]
fireforest_test <- fireforest_norm[415:517, ]

## Training a model on the data ----
# train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# simple ANN with only a single hidden neuron
fireforest_model <- neuralnet(formula = area ~ month + day + FFMC +  DMC  +  DC +  ISI + temp + RH +     
                            wind  + rain  + dayfri + daymon + daysat + daysun + daythu +
                            daytue + daywed + monthapr + monthaug + monthdec + monthfeb + monthjan + monthjul + 
                            monthjun + monthmar + monthmay + monthnov + monthoct + monthsep
                            ,data = fireforest_train)


# visualize the network topology
plot(fireforest_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL
# Re-arange the columns of fireforest_test

fireforest_test <- fireforest_test[,c(11,1:10,12:30)]

results_model <- compute(fireforest_model, fireforest_test[2:30])
# obtain predicted area values
str(results_model)
predicted_area <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_area, fireforest_test$area)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
startups_model2 <- neuralnet(formula = area ~ month + day + FFMC +  DMC  +  DC +  ISI + temp + RH +     
                               wind  + rain  + dayfri + daymon + daysat + daysun + daythu +
                               daytue + daywed + monthapr + monthaug + monthdec + monthfeb + monthjan + monthjul + 
                               monthjun + monthmar + monthmay + monthnov + monthoct + monthsep
                             ,data = fireforest_train, hidden = 3)



# plot the network
plot(startups_model2)

# evaluate the results as we did before
model_results2 <- compute(startups_model2, fireforest_test[2:30])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, fireforest_test$area)

############################## problem3 ##########################################
##### Neural Networks 
library(readr)
# Load the dataset
concrete_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Neural Network\\concrete.csv")
colnames(concrete_data)
#EDA
# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete_data, normalize))

# create training and test data
concrete_train <- concrete_norm[1:825, ]
concrete_test <- concrete_norm[826:1030, ]

## Training a model on the data ----
# train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age ,data = concrete_train)

# visualize the network topology
plot(concrete_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(formula = strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age ,data = concrete_train, hidden = 5)


# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

############################### problem4 #############################################
##### Neural Networks 
library(readr)
# Load the dataset
RPL_data <- read.csv("E:\\DATA SCIENCE ASSIGNMENT\\Class And Assignment Dataset\\Asss\\Neural Network\\RPL.csv")
colnames(RPL_data)
#EDA
#delete state column
RPL_data <- RPL_data[,c(-1,-2,-3,-5,-6)]

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
RPL_norm <- as.data.frame(lapply(RPL_data, normalize))

# create training and test data
RPL_train <- RPL_norm[1:8000, ]
RPL_test <- RPL_norm[8001:10000, ]

## Training a model on the data ----
# train the neuralnet model
install.packages("neuralnet")
library(neuralnet)

# simple ANN with only a single hidden neuron
RPL_model <- neuralnet(formula = Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember  + EstimatedSalary,data = RPL_train)


# visualize the network topology
plot(RPL_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(RPL_model, RPL_test[1:8])
# obtain predicted Exited values
str(results_model)
predicted_Profit <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_Profit, RPL_test$Exited)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
startups_model2 <- neuralnet(formula = Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember  + EstimatedSalary,data = RPL_train,linear.output = F, hidden = 10)

# plot the network
plot(startups_model2)

# evaluate the results as we did before
model_results2 <- compute(startups_model2, RPL_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, RPL_test$Exited)

################################### END ##############################################

