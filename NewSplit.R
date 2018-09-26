#Uploading Packages and Libraries
library(tm) #Using Text Mining Library
#install.packages("glmnet",repos="http://cran.us.r-project.org") #Installing glmnet Packages
library(glmnet) #Using glmnet Library
library(plyr)
library(InformationValue)
library(accSDA)
library(ggplot2)

###############INPUTTING DATA: TRAINING AND TESTING DATA######################
#Reading Data
txt <- read.csv('OriginaljusttweetsJR12s_EDITED.csv')
tweets <- Corpus(DataframeSource(txt))
tweets <-tm_map(tweets,stemDocument) #Removing Stop Words
tweets <-tm_map(tweets,removePunctuation) #Removing Punctuation
dtm <-DocumentTermMatrix(tweets) #Create Document-Term Matrix.
x=as.matrix(dtm) #Setting x as the Document-Term Matrix
trms = colnames(x) #Extracting Column Names/Dictionary


# Set up labels.
Truelabels <-read.csv('OnesandTwos_Corrected.csv')
Truelabels <- data.frame(Truelabels)
m = dim(x)[1]
for (i in 1:m){
  if (Truelabels[i,1] == 1)
    Truelabels[i,1]=0
  if (Truelabels[i,1] == 2)
    Truelabels[i,1]=1
}

# Sample training observations with replacement.
set.seed(55) # Gives relatively balanced sets. Try random sampling with replacement later.

numTrainPerClass <- 250
int <- c(sample(which(Truelabels==0),numTrainPerClass,replace=TRUE), sample(which(Truelabels==1),numTrainPerClass,replace=TRUE))
Training <- x[int,]

# Sample testing observations with replacement too.
numTestPerClass <- 50
tstinds <- c(sample(which(Truelabels==0),numTestPerClass,replace=TRUE), sample(which(Truelabels==1),numTestPerClass,replace=TRUE))
Testing <-x[tstinds,]

# Training labels.
Truelabels_TrainingData <- Truelabels[int,]
Truelabels_TrainingData <- as.numeric(Truelabels_TrainingData)
Truelabels_TrainingData <-as.matrix(Truelabels_TrainingData)
# Testing labels.
Truelabels_TestingData <- Truelabels[tstinds,]
Truelabels_TestingData <- as.numeric(Truelabels_TestingData)
Truelabels_TestingData <- as.matrix(Truelabels_TestingData)
TrainingData = data.frame(cbind(Training,Truelabels_TrainingData))
count(Truelabels_TrainingData == 1)

IndicatorMatrix <-read.csv('Indicator Matrix_TrainingData.csv')
TrainingIndicatorMatrix <- IndicatorMatrix[int,]
TestingIndicatorMatrix <- IndicatorMatrix[tstinds,]
############################################################################################################################################################




###############################################################
###Cross-Validation with Lambda and Gamma###

# Read indicator matrix.
#TrainingIndicatorMatrix <- read.csv('Indicator Matrix_TrainingData.csv')
Labels <- TrainingIndicatorMatrix
nfolds <- 10
n <- length(int) #was 1543
# Make y.
y <- as.matrix(TrainingIndicatorMatrix)

# Define sets of parameters to train.
lams=c(0.05, 0.03, 0.01, 0.001, 0.0001)
gams=c(0.001, 0.0001, 0.01)
nlam <-length(lams)
ngam <- length(gams)

# Number of classes.
K <-2

# Initialize three dimensional array to store misclassification rate for each (fold, lambda, gamma) triple.
mcresults = array(0, c(nlam, ngam, nfolds))
#mcresults_both = array(0, c(nlam, ngam, nfolds))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CV scheme.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for (f in 1:nfolds){

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Set up fold.
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  print(f)

  # Set training/validation observations.
  validation_inds =  seq(floor(numTrainPerClass/nfolds)*(f-1)+1,floor(numTrainPerClass/nfolds)*f, by=1)
  validation_inds = c(validation_inds, validation_inds+numTrainPerClass)

  train_inds = setdiff(seq(1,n, by=1), validation_inds)

  # Extract training and validation observations.
  xtrain = data.frame(Training[train_inds, ])

  # Get problem dimension.
  p = dim(xtrain)[2]

  # Use complement as validation observations.
  xvalidation = data.frame(Training[validation_inds,])

  # Split labels and indicator matrices.
  ytrain = data.frame(y[train_inds, ])
  yvalidation = data.frame(Labels[validation_inds,])
  trainLabels = data.frame(Truelabels_TrainingData[train_inds])
  valLabels = data.frame(Truelabels_TrainingData[validation_inds,])

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Train SDA models for each regularization parameter pair.
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (ll in 1:nlam){
    for (gg in 1:ngam){
      print(lams[ll])
      print(gams[gg])

      # Calculate discriminant vectors for (ll, gg) pair (using SDAAP)
      reslg <- ASDA(as.matrix(xtrain), as.matrix(ytrain), Om=diag(p), gam=gams[gg], lam=lams[ll], q = K-1, method ="SDAAP")

      # Make predictions using nearest centroid rule.
      preds_training <- predict(object=reslg, newdata=as.matrix(xtrain))
      preds_validation <- predict(object = reslg, newdata = as.matrix(xvalidation))

      # Extract predicted scores.
      Predictions_training <- preds_training$x
      Predictions_validation <- preds_validation$x

      # Calculate threshold for classification.
      thresh <- optimalCutoff(actuals=trainLabels, predictedScores = Predictions_training, optimiseFor='misclasserror')
      #thresh_both <- optimalCutoff(actuals=trainLabels, predictedScores = Predictions_training, optimiseFor='Both')

      # Calculate number of validation misclassifications using the predicted scores.
      mcresults[ll, gg, f] = misClassError(actuals = valLabels, predictedScores = Predictions_validation, threshold = thresh) #Changed PredictedScores to preds_validation
      #mcresults_both[ll, gg, f] = misClassError(actuals = valLabels, predictedScores = Predictions_validation, threshold = thresh_both) #Changed PredictedScores to preds_validation

    }
  }
}


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate total misclassification rates and best regularization parameters.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Initialize totals.
TotalMisclass <- 0

# Sum per-fold errors to get total.
for (f in 1:nfolds){
  TotalMisclass = TotalMisclass + mcresults[,,f]
}
print('TotalMisClassification')
print(TotalMisclass)

# Extract indices of parameters resulting in minimum cv-error.
min_TotalMisclass <- which(TotalMisclass == min(TotalMisclass), arr.ind = TRUE)

# Extract best lambda (and display)
print('Best Lambda')
bestl = lams[min_TotalMisclass[1]]
print(bestl)

# Find best gamma (and display)
print('Best Gamma')
bestg = gams[min_TotalMisclass[2]]#was gams[3] ***Gamma is columns***
print(bestg)

# Train optimal scoring discriminant model using optimized regularization parameters.
best_SDAAP = ASDA(as.matrix(Training), as.matrix(y), gam = bestg, lam = bestl, q=K-1, method ="SDAAP" )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate in-sample error.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Make predictions using optimized SOS model.
preds_best_SDAAP <- predict(object=best_SDAAP, newdata=as.matrix(Training))
levels(preds_best_SDAAP$class) = c("0", "1")
lbls = c(rep(1, numTrainPerClass), rep(2, numTrainPerClass))

# Misclassification rate.
count(as.numeric(preds_best_SDAAP$class) == lbls)
err_in = sum(as.numeric(preds_best_SDAAP$class) == lbls)/length(lbls)
misclass_in = 1-err_in

# Make confusion matrix.
confmat <- matrix(0,2,2)
confmat[1,1] = sum(as.numeric((as.numeric(preds_best_SDAAP$class) == 1) & (lbls == 1))) # both 1.
confmat[1,2] = sum(as.numeric((as.numeric(preds_best_SDAAP$class) == 1) & (lbls == 2))) # A2 L1
confmat[2,1] = sum(as.numeric((as.numeric(preds_best_SDAAP$class) == 2) & (lbls == 1))) # A1 L2
confmat[2,2] =  sum(as.numeric((as.numeric(preds_best_SDAAP$class) == 2) & (lbls == 2))) # A2 L1



# #######################################################################
# #          Optimizing for Misclassification Error
# #######################################################################
# # Calculate optimal threshold.
# Threshold_best_SDAAP_INSAMPLE <- optimalCutoff(actuals = Truelabels_TrainingData, predictedScores = preds_best_SDAAP$x, optimiseFor='misclasserror') #predictedScores changed from preds_best_SDAAP$x
# Threshold_best_SDAAP_INSAMPLE <- optimalCutoff(actuals = as.factor(Truelabels_TrainingData), predictedScores = -1*preds_best_SDAAP$x, optimiseFor='misclasserror') #predictedScores changed from preds_best_SDAAP$x
# print('Threshold_best_SDAAP_INSAMPLE')
# print(Threshold_best_SDAAP_INSAMPLE)
# 
# # Create truth table for training data.
# TruthTable_best_SDAAP_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData, predictedScores = preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
# print('TruthTable_best_SDAAP_INSAMPLE')
# print(TruthTable_best_SDAAP_INSAMPLE)
# 
# #Misclassification Error for training.
# MCError_best_SDAAP_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
# print('MCError_best_SDAAP_INSAMPLE')
# print(MCError_best_SDAAP_INSAMPLE)
# 
# ########################################
# # Optimizing for Both
# ########################################
# Threshold_best_SDAAP_both_INSAMPLE <- optimalCutoff(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, optimiseFor='Both') #predictedScores changed from preds_best_SDAAP$x
# print('Threshold_best_SDAAP_both_INSAMPLE')
# print(Threshold_best_SDAAP_both_INSAMPLE)
# 
# TruthTable_best_SDAAP_both_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_both_INSAMPLE) # Uses scores for classification.
# print('TruthTable_best_SDAAP_both_INSAMPLE')
# print(TruthTable_best_SDAAP_both_INSAMPLE)
# 
# MCError_best_SDAAP_both_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_both_INSAMPLE) # Uses scores for classification.
# print('MCError_best_SDAAP_both_INSAMPLE')
# print(MCError_best_SDAAP_both_INSAMPLE)
# 
# #***********************************************************************************
# #     Using Mean as Threshold
# #***********************************************************************************
# #Threshold
# Mean_Threshold_best_SDAAP_INSAMPLE = mean(-1*preds_best_SDAAP$x)
# print('Mean_Threshold_best_SDAAP_INSAMPLE')
# print(Mean_Threshold_best_SDAAP_INSAMPLE)
# 
# #TruthTable
# Mean_TruthTable_best_SDAAP_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
# print('Mean_TruthTable_best_SDAAP_INSAMPLE')
# print(Mean_TruthTable_best_SDAAP_INSAMPLE)
# 
# #Misclassification Error
# Mean_MCError_best_SDAAP_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
# print('Mean_MCError_best_SDAAP_INSAMPLE')
# print(Mean_MCError_best_SDAAP_INSAMPLE)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate out-of-sample error.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Make predictions for testing data using trained model.
Testing_preds_best_SDAAP <- predict(object=best_SDAAP, newdata=as.matrix(Testing))
tstlbls = c(rep(1, numTestPerClass), rep(2, numTestPerClass))
count(as.numeric(Testing_preds_best_SDAAP$class) == tstlbls)
err_out = sum(as.numeric(Testing_preds_best_SDAAP$class) == tstlbls)/length(tstlbls)
misclass_out = 1 - err_out

# Make confusion matrix.
tstconf <- matrix(0,2,2)
tstconf[1,1] = sum(as.numeric((as.numeric(Testing_preds_best_SDAAP$class) == 1) & (tstlbls == 1))) # both 1.
tstconf[1,2] = sum(as.numeric((as.numeric(Testing_preds_best_SDAAP$class) == 1) & (tstlbls == 2))) # A2 L1
tstconf[2,1] = sum(as.numeric((as.numeric(Testing_preds_best_SDAAP$class) == 2) & (tstlbls == 1))) # A1 L2
tstconf[2,2] = sum(as.numeric((as.numeric(Testing_preds_best_SDAAP$class) == 2) & (tstlbls == 2))) # A2 L1
print(tstconf)

# Change class labels to (0,1).
levels(Testing_preds_best_SDAAP$class) = c("0", "1")

#Rounding Predictions to 0 and 1. NEED FOR PLOT
Testing_Predictions_best_SDAAP <- Testing_preds_best_SDAAP$class
Testing_Predictions_best_SDAAP = as.numeric(levels(Testing_Predictions_best_SDAAP))[as.integer(Testing_Predictions_best_SDAAP)]
Testing_Predictions_best_SDAAP = as.matrix(Testing_Predictions_best_SDAAP)

# #########################################
#  #Optimizing for Misclassification Error
# #########################################
# # Calculate optimal threshold (w.r.t. scores.) *Optimizing for Misclassification Error
# Threshold_best_SDAAP_Testing <- optimalCutoff(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, optimiseFor='misclasserror')
# print('Threshold_best_SDAAP_Testing')
# print(Threshold_best_SDAAP_Testing)
# 
# ####TruthTable####
# TruthTable_best_SDAAP_Testing <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing)
# print('TruthTable_best_SDAAP_Testing')
# print(TruthTable_best_SDAAP_Testing)
# 
# ####Misclassification Rate####
# MCError_best_SDAAP_Testing = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing)
# print('MCError_best_SDAAP_Testing')
# print(MCError_best_SDAAP_Testing)
# 
# 
# #########################################
#         #Optimizing for Both
# #########################################
# #Threshold Optimizing for Both
# Threshold_best_SDAAP_Testing_both <- optimalCutoff(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, optimiseFor='Both')
# print('Threshold_best_SDAAP_Testing_both')
# print(Threshold_best_SDAAP_Testing_both)
# 
# # Truthtable.
# TruthTable_best_SDAAP_Testing_both <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing_both)
# print('TruthTable_best_SDAAP_Testing_both')
# print(TruthTable_best_SDAAP_Testing_both)
# 
# #Misclassification Error
# MCError_best_SDAAP_Testing_both = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing_both)
# print('MCError_best_SDAAP_Testing_both')
# print(MCError_best_SDAAP_Testing_both)
# 
# 
# #***********************************************************************************
# #     Using Mean as Threshold
# #***********************************************************************************
# 
# # Calculate optimal threshold (w.r.t. scores.)
# Mean_Threshold_best_SDAAP_Testing <- mean(-1*Testing_preds_best_SDAAP$x)
# print('Mean_Threshold_best_SDAAP_Testing')
# print(Mean_Threshold_best_SDAAP_Testing)
# 
# ####TruthTable####
# Mean_TruthTable_best_SDAAP_Testing <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_Testing)
# print('Mean_TruthTable_best_SDAAP_Testing')
# print(Mean_TruthTable_best_SDAAP_Testing)
# 
# ####Misclassification Rate####
# Mean_MCError_best_SDAAP_Testing = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_Testing)
# print('Mean_MCError_best_SDAAP_Testing')
# print(Mean_MCError_best_SDAAP_Testing)
