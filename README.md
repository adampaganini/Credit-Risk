# Credit-Risk Analysis

## Overview of the Analysis

The purpose of the analysis is to use a machine learning model to accurately predict the number of high-risk and healthy loans. The data set from the lending_data.csv file is financial information that we will use to train a model and then evaluate it. We are splitting the data into training data (to fit our model to) and testing data (to evaluate the accuracy of our model). Each of the training and testing data sets are then split into features and targets.  The target that we are predicting in both the training and testing data sets is the `loan_status` column of data. The features that we are using to make those predictions about `loan_status` include every column (`loan_size`,`interest_rate`,`borrower_income`,`debt_to_income`,`num_of_accounts`,`derogatory_marks`,`total_debt`). We have thus split our original data set into four data frames; training features (`X_train`), testing features (`X_test`), training target (`y_train`), and testing target (`y_test`).  We fit a `LogisticRegression` model around our data, which created a prediction based on our training data (`y_pred`). The accuracy of the prediction was evaluated using `balanced_accuracy_score(y_test, y_pred)`, which compares our `y_test` data against our `y_pred` data and got an accuracy of 89%. Although the accuracy of the overall model was high, the accuracy was subpar for predicting high-risk loans; arguably a more important risk factor when a business is considering handing out loans. The driver of this low accuracy with high-risk loan predictions was due to the data set not being well balanced; by a 2-to-1 margin.  In order to increase the balance of the model we re-sampled the data with a `RandomOverSampler`, which creates more data points for us to sample from.

## Results

* Machine Learning Model 1:
  * Model 1 `clf` overall accuracy = 89%
            precision = 88%
            recall = 89%



* Machine Learning Model 2, after resampling occurred with `RandomOverSampler`:
  * Model 2 `clf_res` overall accuracy = 94%
            precision = 91%
            recall = 94%

## Summary

Here, we recommend using the second machine learning model `clf_res`, which includes an elimination of duplicated rows of data as well as an over-sampled methodology to balance the ratio of healthy loans (0) and high-risk loans (1). The performance one requires depends on the importance of predicting risky loans rather than healthy loans.  The `clf_res` model increases the accuracy of predicting high-risk loans, thus is important for financial institutions to use for protecting the company against handing out loans that won't likely be re-paid.