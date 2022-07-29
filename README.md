# machine failure's analysis
Data science project using Python. Analyze several condition that make the machine tend to be failured and build the model that can identify whether or not the machine were failured. Later on this can be used to do maintenance prediction for the machine that we predicted to be failured.

Objective:

identify what are the most frequent type of failure occurs and why it happens
determine the correlation between type of machine with the occurance of the failure case
determine which condition (features) that tend to make the machine failure
compare several model (SVM, Logistic Regression, KNN, and Decision Tree) to see how accurate it is in predicting failured.

Dataset

Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, this data present and provide a synthetic dataset that reflects real predictive maintenance encountered in the industry. This dataset was downloaded from the UCI Machine Learning Repository with title AI4I 2020 Predictive Maintenance Dataset. 

The dataset consists of 10 000 data points stored as rows with 14 features in columns

- UID: unique identifier ranging from 1 to 10000
- productID: consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial     number
- air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- rotational speed [rpm]: calculated from powepower of 2860 W, overlaid with a normally distributed noise
- torque [Nm]: torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
  'machine failure' label that indicates, whether the machine has failed in this particular data point for any of the following failure modes are true.
- Target : Failure or Not
- Failure Type : Type of Failure

Note:

All features are unscaled and there were some miss imputed on the target's feature, there are some machine that were failure imputed as not failure, so we need to crosscheck it to the failure's type features.
There are no "Null" values, so we don't have to work on ways to replace values.
Most of the case were no failure (96.5 %) of the time, while failure occurs (3.5 %) of the time in the dataframe
