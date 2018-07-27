#Uploading Packages and Libraries
library(tm) #Using Text Mining Library
#install.packages("glmnet",repos="http://cran.us.r-project.org") #Installing glmnet Packages
library(glmnet) #Using glmnet Library
library(plyr)
library(InformationValue)
library(accSDA)
library(ggplot2)

# *****************************************************************************************************************
#                     INPUTTING DATA: TRAINING AND TESTING DATA
# *****************************************************************************************************************
#Reading Data
txt <- read.csv('OriginaljusttweetsJR12s_EDITED.csv')
tweets <- Corpus(DataframeSource(txt))
tweets <-tm_map(tweets,stemDocument) #Removing Stop Words
tweets <-tm_map(tweets,removePunctuation) #Removing Punctuation
dtm <-DocumentTermMatrix(tweets) #Create Document-Term Matrix.
x=as.matrix(dtm) #Setting x as the Document-Term Matrix
trms = colnames(x) #Extracting Column Names/Dictionary

#Read/Process Labels
Truelabels <-read.csv('OnesandTwos_Corrected.csv')
Truelabels <- data.frame(Truelabels)
y = as.matrix(Truelabels)

#Rounding Labels to 0's and 1's
m=1543
Truelabels_0and1 <-vector(mode='numeric', length=m) #create empty vector
Truelabels_0and1=y

for (i in 1:m){
  if (Truelabels_0and1[i] == 1)
    Truelabels_0and1[i]=0
  if (Truelabels_0and1[i] == 2)
    Truelabels_0and1[i]=1
}

#Splitting Observations into 80/20-Split
set.seed(55)
int = sort(sample(nrow(x), nrow(x)*.80))
TrainingData <- x[int,]
TestingData <-x[-int,]

#Training and Testing Labels for 80/20-Split of Observations
TrainingLabels <- Truelabels_0and1[int,]
TestingLabels <-Truelabels_0and1[-int,]

# ******************************************Resampling*************************************************
#                    Resampling Training Data with Replacement
# *****************************************************************************************************
#Sampling Training Observations with Replacement
#True Labels for Sampled Training Data and Training Observations
numTrainPerClass <- 400 # was617
trainint <- c(sample(which(TrainingLabels==0), numTrainPerClass, replace= TRUE), sample(which(TrainingLabels==1), numTrainPerClass, replace=TRUE))
Truelabels_TrainingData_0and1 <- TrainingLabels[trainint]

#Training Labels as a matrix
Truelabels_TrainingData <- Truelabels_TrainingData_0and1
Truelabels_TrainingData <- as.numeric(Truelabels_TrainingData)
Truelabels_TrainingData <-as.matrix(Truelabels_TrainingData)

#Training Observations
TrainingData <- TrainingData[trainint,]

#Testing Labels as a Matrix
Truelabels_TestingData <- TestingLabels
Truelabels_TestingData <- as.numeric(Truelabels_TestingData)
Truelabels_TestingData <- as.matrix(Truelabels_TestingData)

#Testing Observations
TestingData <- TestingData

TrainingData = data.frame(cbind(TrainingData,Truelabels_TrainingData_0and1))


#Converting to Matrices for Subsetting
dtm.mat <- as.matrix(TrainingData) #Training Data
Test_dtm.mat <- as.matrix(TestingData) #Testing Data
Test_dframe <- data.frame(Test_dtm.mat)
xx <-data.frame(Test_dtm.mat[,intersect(colnames(Test_dtm.mat),colnames(dtm.mat))]) #Subsetting Testing Data by Column Names (Terms)
yy <- read.table(textConnection(""),col.names=colnames(dtm.mat),colClasses="integer") #Making an empty data frame with the Column Names of the Training Data

#Configuring Indicator Matrix
IndicatorMatrix <-read.csv('Indicator Matrix_TrainingData.csv')
TrainingIndicatorMatrix <- IndicatorMatrix[trainint,]
TestingIndicatorMatrix <- IndicatorMatrix[TestingData,]

############################################################################################################################################################

# Illustrates issue with indicator matrix.
err = cbind(cbind(TrainingIndicatorMatrix, Truelabels_TrainingData) )

            