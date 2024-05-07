rm(list=ls())

# Package for Distributed Random Forests
library(drf)    ## distributional random forest
# Package for evaluating probabilistic forecasts: crps_sample
library(scoringRules) ### CRPS
#Package for  data normalization
# preProcess(x)
library(caret)      
# Package for Distributed k-Nearest Neighbors
library(KernelKnn)     

## choice of data set
load("qsar.Rda") 
load("airfoil.Rda") 
#data_set <- qsar
data_set <- airfoil
# Length  samples
n <- nrow(data_set)
# Number of features (excluding the target variable)
p <- ncol(data_set)-1

set.seed(123)
# Number of repetitions for the experiment
Nrep <- 100
## Matrix to store model selection errors
errorSelect <- matrix(NA,  nrow = Nrep, ncol = 2)
# Matrix to store CRPS errors for each model
ErrorL2              <- matrix(NA,  nrow = Nrep, ncol = 4)
# Column names for ErrorL2 matrix
colnames(ErrorL2)  <- c( "DRF", "KNN", "MS", "CA") 
# Marking the start time of the experiment
start.time <- Sys.time()
for(ii in 1:Nrep){
  cat(ii, " ")
  ## training / validation / test
  # Proportions for training, validation, and test sets
  proba <- c(0.5,0.2,0.3)
  # Randomly shuffle the indices of the dataset
  random_order <- sample(n,n)
  # Splitting indices for training, validation, and test sets based on proportions
  train <- random_order[1:ceiling(proba[1]*n)]
  val <- random_order[ceiling(proba[1]*n+1):ceiling((proba[1]+proba[2])*n)]
  test <- random_order[ceiling((proba[1]+proba[2])*n+1):n]
  # Training data
  # Target variable for training
  Y_train <- data_set[train,p+1]
  # Features for training
  X_train <- data_set[train,1:p]
  # Preprocessing object
  center_scale <- preProcess(X_train)
  # Applying preprocessing to training features
  X_train <- predict(center_scale,X_train)
  # Validation data
 # Target variable for validation
  Y_val <- data_set[val,p+1]
  # Features for validation
  X_val <- data_set[val,1:p]
  # Applying same preprocessing to validation features
  X_val <- predict(center_scale,X_val)
  # Test data 
  # Target variable for test
  Y_test <- data_set[test,p+1]
  # Features for test
  X_test <- data_set[test,1:p]
  # Applying same preprocessing to test features
  X_test <- predict(center_scale,X_test)
  
  
  ########### drf Xval
  ### Function for computing validation error for Distributed Random Forest (DRF) with different mtry values
  ### choose mtry
  val_error_DRF <- function(m){
    fit_DRF <- drf(X=X_train,Y=Y_train,mtry=m,num.trees=1000,
                   sample.fraction=0.9,ci.group.size = 1,min.node.size=1)
    weight_val <- as.matrix(get_sample_weights(fit_DRF,X_val))
    crps_val_DRF <- mean(crps_sample(
      Y_val, 
      matrix(Y_train,ncol=length(Y_train),nrow=length(Y_val),byrow=T), 
      w=weight_val))
    return(crps_val_DRF)
  }
  # Grid of mtry values to choose from
  m <- 1:p
  #Computing validation error for each mtry value using the val_error_DRF function
  result_val_error_DRF <- sapply(1:p,val_error_DRF)
  # Selecting the optimal mtry value that minimizes the validation error
  choosem <-m[which.min(result_val_error_DRF)]  
  # Printing the optimal mtry value
#  print(choosem)
  # Calculating the minimum validation error (CRPS) for the chosen mtry
  crps_val_DRF <- min(result_val_error_DRF)
  
  # Function for computing validation error for k Nearest Neighbors (KNN) with different k values
  # choose the best k
  val_error_KNN <- function(l){
    knn_idx <- knn.index.dist(X_train,X_val,k=l)[[1]]
    weight_val <- matrix(0,nrow=length(Y_val),ncol=length(Y_train))
    for (i in 1:length(Y_val)){
      weight_val[i,knn_idx[i,]] <- 1/l
    }
    crps_val_KNN <- mean(crps_sample(
      Y_val, 
      matrix(Y_train,ncol=length(Y_train),nrow=length(Y_val),byrow=T), 
      w=weight_val))
    return(crps_val_KNN)
  }
  # Range of k values to choose from
  l <- 1:(nrow(X_train)-1)
  # Computing validation error for each k value using the val_error_KNN function
  result_val_error_KNN <- sapply(l,val_error_KNN)
  # Selecting the optimal k value that minimizes the validation error
  choosek <-l[which.min(result_val_error_KNN)]
 # print(choosek)
  # Calculating the minimum validation error (CRPS) for the chosen k value
  crps_val_KNN <- min(result_val_error_KNN)
  # Storing the validation errors of DRF and KNN in the errorSelect matrix
  errorSelect[ii,]        <- c(crps_val_DRF,crps_val_KNN)  
  # Selecting the model (DRF or KNN) with the minimum validation error
  idxMS                   <- which.min(errorSelect[ii, ])  ##MS Selection##
  print(idxMS)
  #####
                           ############ convex aggregation
  # Function for computing convex aggregation of DRF and KNN models
  convex_risk <- function(beta){
    # Normalization of lambdas
    lamb              <- exp(beta)/sum(exp(beta)) 
    # DRF model
    fit_DRF <- drf(X=X_train,Y=Y_train,mtry=choosem,num.trees=1000,
                   sample.fraction=0.9,ci.group.size = 1,min.node.size=1)
    weight_valdrf <- as.matrix(get_sample_weights(fit_DRF,X_val))
    # KNN model
    knn_idx <- knn.index.dist(X_train,X_val,k=choosek)[[1]]
    weight_valknn <- matrix(0,nrow=length(Y_val),ncol=length(Y_train))
    for (i in 1:length(Y_val)){
      weight_valknn[i,knn_idx[i,]] <- 1/choosek
    }
    # Combining weights of DRF and KNN models using lambdas
    weight_knn_drf <- weight_valdrf*lamb[1]+weight_valknn*lamb[2] 
    # Computing the mean CRPS using the combined weights
    out <- mean(crps_sample(
      Y_val, 
      matrix(Y_train,ncol=length(Y_train),nrow=length(Y_val),byrow=T), 
      w=weight_knn_drf))
    return(out)
  }
  # Initializing lambdas for convex aggregation
  lamb_initial <- rep(0,2)
  # Finding the optimal lambdas using optimization method
  optimal_lambda    <- optim(lamb_initial, convex_risk) #, method = "L-BFGS-B", control = list(maxit = 50000000)) #BFGS
  # Calculating the best lambdas after normalization
  hatlamb           <- exp(optimal_lambda$par)/sum(exp(optimal_lambda$par)) 
  # Printing the best lambda values
  print(hatlamb)
  
  ########## Union of two samples
  X_train_val              <- rbind(X_train,X_val) 
  Y_train_val              <- c(Y_train,Y_val)  
  
  ####### DRF
  # Training a DRF model using the combined training and validation data
  fit_DRF <- drf(X=X_train_val,Y=Y_train_val,mtry=choosem,num.trees=1000,
                 sample.fraction=0.9,ci.group.size = 1,min.node.size=1)
  # Computing sample weights for the test data using the trained DRF model
  weight_val_train <- as.matrix(get_sample_weights(fit_DRF,X_test))
  # Computing CRPS for DRF model using the test data
  crps_train_val_DRF <- mean(crps_sample(
    Y_test, 
    matrix(Y_train_val,ncol=length(Y_train_val),nrow=length(Y_test),byrow=T), 
    w=weight_val_train))  
  # Storing the CRPS value for DRF model in the ErrorL2 matrix
  ErrorL2[ii,1] <- crps_train_val_DRF
  
  ###### knn
  # Finding the k-nearest neighbors indices for the test data using the combined training and validation data
  knn_idx <- knn.index.dist(X_train_val,X_test,k=choosek)[[1]]
  # Initializing weights for the test data using the k-nearest neighbors
  weight_val_trainknn <- matrix(0,nrow=length(Y_test),ncol=length(Y_train_val))
  for (i in 1:length(Y_test)){
    weight_val_trainknn[i,knn_idx[i,]] <- 1/choosek
  }
  # Computing CRPS for KNN model using the test data
  crps_val_trainKNN <- mean(crps_sample(
    Y_test, 
    matrix(Y_train_val,ncol=length(Y_train_val),nrow=length(Y_test),byrow=T), 
    w=weight_val_trainknn))
  # Storing the CRPS value for KNN model in the ErrorL2 matrix
  ErrorL2[ii,2] <- crps_val_trainKNN
                     # MS (Model Selection)
  # Selecting the CRPS value based on the model with the minimum validation error (DRF or KNN)
  Msmodel        <- c(crps_train_val_DRF,crps_val_trainKNN)[idxMS]
  # Storing the selected CRPS value in the ErrorL2 matrix
  ErrorL2[ii,3]  <-  Msmodel
                         # Convex Aggregation
  # Computing weights for convex aggregation using the optimal lambdas
  weight_bestLamb <- weight_val_train*hatlamb[1]+ weight_val_trainknn*hatlamb[2]
  # Computing the mean CRPS using the weights for convex aggregation
  #  # Storing the CRPS value for KNN model in the ErrorL2 matrix
  ErrorL2[ii,4]  <-  mean(crps_sample(
    Y_test, 
    matrix(Y_train_val,ncol=length(Y_train_val),nrow=length(Y_test),byrow=T), 
    w=weight_bestLamb))
}
save(ErrorL2,file="result_AirfoilNelder_Mead.Rda")
cat("\n", "resultats obtenus", "\n")
# Computing and printing the mean of CRPS errors for each model
print(round(apply(ErrorL2,2,mean), digits =3))
# Computing and printing the standard deviation of CRPS errors for each model
print(round(apply(ErrorL2,2,sd)/10, digits = 3))
# Marking the end time of the experiment
end.time <- Sys.time()
# Calculating the time taken for the experiment to run
time.taken <- round(end.time - start.time,2)
time.taken
# Importing ggplot2 library for data visualization
library(ggplot2)
# Importing reshape2 library for data manipulation
library(reshape2)
ErrorL2 <- ErrorL2[, c(2,1,3,4)]
colnames(ErrorL2)  <- c( "KNN","DRF", "MS", "CA") 
pdf("AirfoilNelder_MeadMSCA.pdf")
# Creating a ggplot object named graphe


graphe <- ggplot(data=melt(data.frame(ErrorL2)),aes(x=variable,y=value, fill = variable))+
  geom_boxplot()+ ylim(min(ErrorL2)-0.02, max(ErrorL2)+0.02)+
  labs(title = "", x="", y= expression("CRPS (test)")) +theme_bw()+
  theme(legend.position = "none", axis.title.y = element_text(size = 25), axis.text.x =element_text(size = 20) , axis.text.y=element_text(size = 20), 
        panel.background = element_rect(fill = "gray90"))

# Displaying the plot
graphe  
dev.off()

