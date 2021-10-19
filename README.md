# R-programming
library("tibble")
library("caret")
library("dplyr")
library("kiaR")
library("naivebayes")

#read Data
train = read.csv("smsspam_train.csv")
test = read.csv("smsspam_test.csv")


# Data prep for logistic and decision tree model
train = as.tibble(smsspam_train)
test = as.tibble(smsspam_test)
train = train %>% mutate(label = as.factor(label))
test = test %>% mutate(label = as.factor(label))

# data prep for Naive Bayes
train_x = as.matrix(train[,c(1:500)])
train_y = as.matrix(train[,c(501)])

test_x = as.matrix(test[,c(1:500)])
train_y = as.matrix(train[,c(501)])

# Naive Bayes Model
nb_model = bernoulli_naive_bayes(train_x, train_y)

#naive bayes prediction variable
nb_predictions = as.numeric(as.character(predict(nb_model, test_x, type = "class")))

#write nb_predictions.txt file
write.table(nb_predictions, "nb_predictions.txt")


## Logistic model
#checking the structure of training dataset
levels(train$label)

# Fitting logistic model
lg_model = glm(label ~., data = train, family = "binomial")

#function for Calculating Accuracy
calc_accuracy = function(actual, predicted){
  mean(actual == predicted)
}

#logistic regression prediction variable
lg_predictions = ifelse(predict(lg_model, test, type = "response") > 0.5, 1, 0)


#checking lg model accuracy
calc_accuracy(test$label, nb_predictions)

#checking lg model accuracy
calc_accuracy(test$label, lg_predictions)

# Writing a function to get the f1 score
# 0 | Negative
# 1 | positive

compute_score = function(pred, val){
  
  # f1_score function takes in two parameters prediction vector and validation dataset
  
  tp = sum(pred == 1 & val$label == 1)
  tn = sum(pred == 0 & val$label == 0)
  fp = sum(pred == 1 & val$label == 0)
  fn = sum(pred == 0 & val$label == 1)
  
  # ppv or precision
  ppv = tp / (tp + fp)
  
  # tpr or recall or sensitivity
  tpr = tp / (tp + fn)
  
  # F1 score is harmonic mean of precision and recall
  f1 = 2 * ((ppv*tpr)/(ppv+tpr))
  
  return(f1)
  
}

# Naive_Bayes F1 Score calculated using compute_score function
compute_score(nb_predictions, test)


#Logistic F1 Score calculated using compute_score function
compute_score(lg_predictions, test)

#F1 score of Naive bayes is better that is 0.9353448 compared to logistic regression that is 0.8622047.
# Naive Bayes performs better on test set

#reading and storing decision tree predictions
pred_dec_tree = dt_predictions[[c("V1")]]

# Decision tree F1 Score calculated using compute_score function
compute_score(pred_dec_tree, test)

#F1 score of decision tree (0.8870637) is better than logistic regression but less than Naive bayes

# Converting all the predictions column from factor to numeric
list_pred = list(lg_predictions, nb_predictions, pred_dec_tree)
pred_df = as.data.frame(list_pred)
#giving column names
colnames(pred_df) = c("logistic_predictions","naive_bayes_predictions","decision_tree_predicitions")


#writing function to calculate majority voting 
vote_function = function(x){
  sum_vote = sum(x["logistic_predictions"] + x["naive_bayes_predictions"] + x["decision_tree_predicitions"])
  if(sum_vote == 3 | sum_vote == 2){
    1
  }else{
    0
  }
}

ensemble_prediction = apply(pred_df, 1, vote_function)

# ensemble prediction F1 Score calculated using compute_score function
compute_score(ensemble_prediction, test)

# ensemble prediction is better that single classifier on the test set
