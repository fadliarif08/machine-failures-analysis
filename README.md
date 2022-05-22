# maintenance-predictions
Data Science project using Python. Create machine learning model (SVM and Logistic Regression) to predict whether or not the machine will failure at certain condition. Later on we can do the maintenance to the machine that we predicted to be failured.

Achieved 86.2 % accuracy on testing set and 93.9 % accuracy on training set using non linear SVM, successfully predicted 84 % of machine that were failure and 88 % of machine were not failure

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
