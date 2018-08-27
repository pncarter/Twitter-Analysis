# ***********************************************************************************************************
#                                          INPUTTING DATA
# ***********************************************************************************************************
source("Autism Data_80_20 Split.R")
library(caret)
###############################################################
###Cross-Validation with Lambda and Gamma###

# Read indicator matrix.
Labels <- TrainingIndicatorMatrix
nfolds <- 10
n <- length(trainint) #was 1543
# Make y.
y <- as.matrix(TrainingIndicatorMatrix)

# Define sets of parameters to train.
lams=c(0.0001, 0.00001, 0.00000001)
gams=c(0.000001, 0.00001, 0.000001)
nlam <-3
ngam <- 3

# Number of classes.
K <-2

# Initialize three dimensional array to store misclassification rate for each (fold, lambda, gamma) triple.
mcresults = array(0, c(nlam, ngam, nfolds))
mcresults_both = array(0, c(nlam, ngam, nfolds))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CV scheme.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for (f in 1:nfolds){

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Set up fold.
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #print(f)

  # Set training/validation observations.
  validation_inds =  seq(floor(numTrainPerClass/nfolds)*(f-1)+1,floor(numTrainPerClass/nfolds)*f, by=1)
  validation_inds = c(validation_inds, validation_inds+numTrainPerClass)
  train_inds = setdiff(seq(1,n, by=1), validation_inds)


  # Extract training and validation observations.
  xtrain = data.frame(TrainingData[train_inds, ])

  # Get problem dimension.
  p = dim(xtrain)[2]

  # Use complement as validation observations.
  xvalidation = data.frame(TrainingData[validation_inds,])

  # Split labels and indicator matrices.
  ytrain = data.frame(Labels[train_inds, ]) ###
  yvalidation = data.frame(Labels[validation_inds,])
  trainLabels = data.frame(Truelabels_TrainingData[train_inds])
  valLabels = data.frame(Truelabels_TrainingData[validation_inds,])

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Train SDA models for each regularization parameter pair.
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (ll in 1:nlam){
    for (gg in 1:ngam){
      #print(lams[ll])
      #print(gams[gg])

      # Calculate discriminant vectors for (ll, gg) pair (using SDAAP)
      reslg <- ASDA(as.matrix(xtrain), as.matrix(ytrain), Om=diag(p), gam=gams[gg], lam=lams[ll], q = K-1, method ="SDAAP", na.action=na.omit)

      # Make predictions using nearest centroid rule.
      preds_training <- predict(object=reslg, newdata=as.matrix(xtrain))
      preds_validation <- predict(object = reslg, newdata = as.matrix(xvalidation))

      # Change predicted classes labels to agree with actual labels.
      levels(preds_training$class) = c("0","1")
      Predictions_training <- preds_training$class # Should be preds_training$scores/values
      #Predictions_training <- preds_training$x
      Predictions_training = as.numeric(levels(Predictions_training))[as.integer(Predictions_training)]

      levels(preds_validation$class) = c("0","1")
      Predictions_validation <- preds_validation$class
      #Predictions_validation <- preds_validation$x
      Predictions_validation = as.numeric(levels(Predictions_validation))[as.integer(Predictions_validation)]

      # Calculate threshold for classification.
      thresh <- optimalCutoff(actuals=trainLabels, predictedScores = Predictions_training, optimiseFor='misclasserror')
      thresh_both <- optimalCutoff(actuals=trainLabels, predictedScores = Predictions_training, optimiseFor='Both')
      #print('thresh')
      #print(thresh)
      #print('thresh_both')
      #print(thresh_both)

      # Calculate number of validation misclassifications using the predicted scores.
      mcresults[ll, gg, f] = misClassError(actuals = valLabels, predictedScores = Predictions_validation, threshold = thresh) #Changed PredictedScores to preds_validation
      mcresults_both[ll, gg, f] = misClassError(actuals = valLabels, predictedScores = Predictions_validation, threshold = thresh_both) #Changed PredictedScores to preds_validation

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

# Change class labels to (1,2).
levels(preds_best_SDAAP$class) = c("0", "1")

#Rounding Predictions to 0 and 1 NEED FOR PLOT
Predictions_best_SDAAP <- preds_best_SDAAP$class
Predictions_best_SDAAP = as.numeric(levels(Predictions_best_SDAAP))[as.integer(Predictions_best_SDAAP)]
Predictions_best_SDAAP = as.matrix(Predictions_best_SDAAP)

#######################################################################
#          Optimizing for Misclassification Error
#######################################################################
# Calculate optimal threshold.
Threshold_best_SDAAP_INSAMPLE <- optimalCutoff(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, optimiseFor='misclasserror') #predictedScores changed from preds_best_SDAAP$x
print('Threshold_best_SDAAP_INSAMPLE')
print(Threshold_best_SDAAP_INSAMPLE)

# Create truth table for training data.
TruthTable_best_SDAAP_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
print('TruthTable_best_SDAAP_INSAMPLE')
print(TruthTable_best_SDAAP_INSAMPLE)

#Misclassification Error for training.
MCError_best_SDAAP_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
print('MCError_best_SDAAP_INSAMPLE')
print(MCError_best_SDAAP_INSAMPLE)

########################################
# Optimizing for Both
########################################
Threshold_best_SDAAP_both_INSAMPLE <- optimalCutoff(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, optimiseFor='Both') #predictedScores changed from preds_best_SDAAP$x
print('Threshold_best_SDAAP_both_INSAMPLE')
print(Threshold_best_SDAAP_both_INSAMPLE)

TruthTable_best_SDAAP_both_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_both_INSAMPLE) # Uses scores for classification.
print('TruthTable_best_SDAAP_both_INSAMPLE')
print(TruthTable_best_SDAAP_both_INSAMPLE)

MCError_best_SDAAP_both_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_both_INSAMPLE) # Uses scores for classification.
print('MCError_best_SDAAP_both_INSAMPLE')
print(MCError_best_SDAAP_both_INSAMPLE)

###ASDA Plot (Training data)###
z_ASDA = preds_best_SDAAP$x
plot(z_ASDA, Predictions_best_SDAAP, ylab = "Classification of best_SDA", xlab = "Prediction Scores", col = c("red", "blue"))


#***********************************************************************************
#     Using Mean as Threshold
#***********************************************************************************
#Threshold
Mean_Threshold_best_SDAAP_INSAMPLE = mean(preds_best_SDAAP$x)
print('Mean_Threshold_best_SDAAP_INSAMPLE')
print(Mean_Threshold_best_SDAAP_INSAMPLE)

#TruthTable
Mean_TruthTable_best_SDAAP_INSAMPLE <-confusionMatrix(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
print('Mean_TruthTable_best_SDAAP_INSAMPLE')
print(Mean_TruthTable_best_SDAAP_INSAMPLE)

#Misclassification Error
Mean_MCError_best_SDAAP_INSAMPLE = misClassError(actuals = Truelabels_TrainingData[,1], predictedScores = -1*preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_INSAMPLE) # Uses scores for classification.
print('Mean_MCError_best_SDAAP_INSAMPLE')
print(Mean_MCError_best_SDAAP_INSAMPLE)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate out-of-sample error.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Make predictions for testing data using trained model.
Testing_preds_best_SDAAP <- predict(object=best_SDAAP, newdata=as.matrix(Testing))

# Change class labels to (0,1).
levels(Testing_preds_best_SDAAP$class) = c("0", "1")

#Rounding Predictions to 0 and 1. NEED FOR PLOT
Testing_Predictions_best_SDAAP <- Testing_preds_best_SDAAP$class
Testing_Predictions_best_SDAAP = as.numeric(levels(Testing_Predictions_best_SDAAP))[as.integer(Testing_Predictions_best_SDAAP)]
Testing_Predictions_best_SDAAP = as.matrix(Testing_Predictions_best_SDAAP)

#########################################
#Optimizing for Misclassification Error
#########################################
# Calculate optimal threshold (w.r.t. scores.) *Optimizing for Misclassification Error
Threshold_best_SDAAP_Testing <- optimalCutoff(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, optimiseFor='misclasserror')
print('Threshold_best_SDAAP_Testing')
print(Threshold_best_SDAAP_Testing)

####TruthTable####
TruthTable_best_SDAAP_Testing <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing)
print('TruthTable_best_SDAAP_Testing')
print(TruthTable_best_SDAAP_Testing)

####Misclassification Rate####
MCError_best_SDAAP_Testing = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing)
print('MCError_best_SDAAP_Testing')
print(MCError_best_SDAAP_Testing)


#########################################
#Optimizing for Both
#########################################
#Threshold Optimizing for Both
Threshold_best_SDAAP_Testing_both <- optimalCutoff(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, optimiseFor='Both')
print('Threshold_best_SDAAP_Testing_both')
print(Threshold_best_SDAAP_Testing_both)

#TruthTable
TruthTable_best_SDAAP_Testing_both <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing_both)
print('TruthTable_best_SDAAP_Testing_both')
print(TruthTable_best_SDAAP_Testing_both)

#Misclassification Error
MCError_best_SDAAP_Testing_both = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Threshold_best_SDAAP_Testing_both)
print('MCError_best_SDAAP_Testing_both')
print(MCError_best_SDAAP_Testing_both)

#Plot of ASDA Resuls (Testing Data)
###ASDA Plot###
z_ASDA_Testing = Testing_preds_best_SDAAP$x
plot(z_ASDA_Testing, Testing_Predictions_best_SDAAP, ylab = "Classification of Testing_best_SDA", xlab = "Testing_Prediction Scores", col = c("red", "blue"))

#***********************************************************************************
#     Using Mean as Threshold
#***********************************************************************************

# Calculate optimal threshold (w.r.t. scores.)
Mean_Threshold_best_SDAAP_Testing <- mean(Testing_preds_best_SDAAP$x)
print('Mean_Threshold_best_SDAAP_Testing')
print(Mean_Threshold_best_SDAAP_Testing)

####TruthTable####
Mean_TruthTable_best_SDAAP_Testing <-confusionMatrix(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_Testing)
print('Mean_TruthTable_best_SDAAP_Testing')
print(Mean_TruthTable_best_SDAAP_Testing)

####Misclassification Rate####
Mean_MCError_best_SDAAP_Testing = misClassError(actuals = Truelabels_TestingData[,1], predictedScores = -1*Testing_preds_best_SDAAP$x, threshold = Mean_Threshold_best_SDAAP_Testing)
print('Mean_MCError_best_SDAAP_Testing')
print(Mean_MCError_best_SDAAP_Testing)
