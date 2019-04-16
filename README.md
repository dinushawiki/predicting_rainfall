# Predicting rainfall in Australia

## Import Data

Can we predict whether it is going to rain tomorrow from weather data obtained today? This dataset contains daily weather observations from numerous Australian weather stations. The target variable RainTomorrow means: Did it rain the next day? Yes or No.

We are going to train diffrent binary classification algorithms from the supervised dataset provided and determine how accurate we can predict if it is going to rain tomorrow given today's weather conditions. Let's begin.

## Data Fields

* Date: The date of observation
* Location: The common name of the location of the weather station
* MinTemp: The minimum temperature in degrees celsius
* MaxTemp: The maximum temperature in degrees celsius
* Rainfall: The amount of rainfall recorded for the day in mm
* Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
* Sunshine: The number of hours of bright sunshine in the day.
* WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
* WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
* WindDir9am: Direction of the wind at 9am
* WindDir3pm: Direction of the wind at 3pm
* WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
* WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3pm
* Humidity9am: Humidity (percent) at 9am
* Humidity3pm: Humidity (percent) at 3pm
* Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
* Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
* Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
* Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
* Temp9am: Temperature (degrees C) at 9am
* Temp3pm: Temperature (degrees C) at 3pm
* RainToday: Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
* RISK_MM: The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".
* RainTomorrow: The target variable. Did it rain tomorrow?

## Explore the Data

Data exploration showed that there are few columns with over 40% empty values. These columns (Sunshine, Evaporation, Cloud3pm, and Cloud9am) are dropped from the analysis. Also Risk_MM indicates  the amount of rainfall in millimeters for the next day. This value is used to determine the target variable "RainTomorrow". So Risk_MM should be ignored here as this would give the model a false accuracy. 
determine the target variable "RainTomorrow". So it should be ignored here as this would give the model a false accuracy. We will fill the missing values of columns that we didn't drop later in the preprocessing pipeline.

Split the data using train_test_split.

```
train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 123)
```

Furthur explorations shows that there are no outliers in the data and all the data points lie within acceptable ranges.
The correlation matrix for the feature varibles and target shown here.

![correlation_matrix](https://user-images.githubusercontent.com/24527000/56228587-29656980-6046-11e9-84f0-de714465be79.png)

From the correlation matrix above RainToday seem to have a very high correlation with RainFall, which make sense as RainFall provides the amount of rain we got today. Also RainToday has a high correlation with many other columns in this data set. So let's drop RainToday column.

First let's generate our categorical variable pipeline. For categorical variables we decieded to change the null values to most frequent values in the column. We also used OneHotEncoder to encode categorical data.

```
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["WindGustDir", "WindDir9am", "WindDir3pm"])),
        ("imp", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
```
For numerical variables we decieded to change the null values to the mean in the column. We also used StandardScaler to normalize numerical values.

```
num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                                              "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"])),
        ("imp", SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ])
```
Crate the train and test sets to feed the classifiers. Here the train and test features are fed through the pipeline for preprocessing and the train and test target column (RainTomorrow) values are mapped to 1 and 0 instead of 'Yes' and 'No'.

```
#train
X_train_prepared = preprocess_pipeline.fit_transform(X)
y_train_prepared = y.map({'Yes':1, 'No':0})

#test
X_test_prepared = preprocess_pipeline.fit_transform(X_test)
y_test_prepared = y_test.map({'Yes':1, 'No':0})
```
## SHORT-LIST PROMISING MODELS and FINE-TUNE THE SYSTEM

We are choosing roc_auc or area under the ROC curve as the metric to measure the performance of each model. AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting roc_auc is as the probability that the model ranks a random positive example more highly than a random negative example. This metric is chosen because we want the model to identify as mamy possible positive cases (rain tomorrow) as possible. Positive case here being it will rain tomorrow and negative case being it will not rain tomorrow.

roc_auc is desirable for the following two reasons:

roc_auc is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
roc_auc is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

### LogisticRegression

The first classifier to test is Logistic Regression. Logistic regression is appropriate to conduct regression analysis when the dependent variable is dichotomous (binary). This dataset is not very large and we are testing for both l1 and l2 penalties. Therefore ‘liblinear’ is used as the solver. We are tuning the parameters C and penalty of the LogisticRegression classifier.

* C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
* penalty: Used to specify the norm used in the penalization.(‘l1’ or ‘l2’)

The optimal parameters obtained from tuning are {'C': 1.0, 'penalty': 'l1'} with a roc_auc value of 0.85495.

We train the LogisticRegression regression model with given parameters and the training set.

```
log_best_model = LogisticRegression(C = 1.0, penalty = 'l1', solver = 'liblinear')
```
We then calculate the roc_auc and plot the ROC curve with the test set.

![roc_log](https://user-images.githubusercontent.com/24527000/56230992-5c5e2c00-604b-11e9-8531-e93f6829c124.png)

LogisticRegression classifier gives a roc_auc (area under the curve) value of 0.85575.

Similar calculations are performed with other classifiers and the results are shown below.

### RandomForestClassifier

A random forest is a classifier that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement when bootstrap=True (default). We are tuning the parameters max_depth and n_estimators of the RandomForestClassifier classifier.

* max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
* n_estimators: The number of trees in the forest.

The optimal parameters obtained from tuning are {'bootstrap': True, 'max_depth': 60, 'n_estimators': 300} with a roc_auc value of 0.87432.

```
rf_best_model = RandomForestClassifier(bootstrap = True, max_depth = 60, n_estimators = 300, random_state=123)
```
![roc_rf](https://user-images.githubusercontent.com/24527000/56231180-d1316600-604b-11e9-8589-62eeb3522ef2.png)

RandomForestClassifier classifier gives a roc_auc value of 0.87900. RandomForestClassifier gives a higher roc_auc than LogisticRegression. 

In addition to classifying RandomForestClassifier can also be used to determine feature importance. Here we plot the significance of each feature when classifying the given features set into positive and negative classes.

### LGBMClassifier

Light GBM is a gradient boosting framework that uses tree based learning algorithm. It grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm. Parameters assigned are,

objective: Specifies the application of your model, whether it is a regression problem or classification problem. This is binary classification problem so 'binary' is assigned.
metric: Specifies loss for model building. 'binary_logloss' is appropriate for loss for binary classification.
boosting: Defines the type of algorithm you want to run. 'dart' is used for better accuracy.
Parameters tuned are,

min_data_in_leaf: Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.
max_depth: The maximum depth of the tree.
learning_rate: This determines the impact of each tree on the final outcome. GBM works by starting with an initial estimate which is updated using the output of each tree. The learning parameter controls the magnitude of this change in the estimates.

The optimal parameters obtained from tuning are {'learning_rate': 0.4, 'max_depth': 10, 'min_data_in_leaf': 300} with a roc_auc value of 0.87731.

```
lgbm_best_model = LGBMClassifier(application = 'binary', metric = 'binary_logloss', boosting = 'dart', min_data_in_leaf = 300, max_depth = 10, learning_rate = 0.4)
```
![roc_lgbm](https://user-images.githubusercontent.com/24527000/56231384-561c7f80-604c-11e9-85be-43bb95842f67.png)

LGBMClassifier classifier gives a roc_auc value of 0.87829. This is slightly less than RandomForestClassifier. However, LGBMClassifier runs faster than RandomForestClassifier.

### Neural Network

Lastly we try a simple Neural network model to predict if it is going to rain tomorrow. Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks. We are using a Sequential Neural Netwrok with nodes and layers. The optimal Neural Network is obtained by chnaging the number of nodes and layers in the model. Then these optimal number of nodes and layers are used to train the Neural network with training data.

First we need to figure out what optimizer to use. Our choices are,

* adam optimizer
* Stochastic gradient descent optimizer with different learning rates.

Running optimizations tests we determined that adam optimizer gives the lowest loss. Therefore we are using the 'adam' optimizer in our deep learning model. Now we need to train the model and validate it. We will increase the number of nodes and layers to get the best posiible validation score possible.

Running the validation tests we determined that increasing the number of nodes to 120 decreased the loss of the model. Therefore we are going to use 120 as the number of nodes. Increasing the number of layers did not decrease the loss of the model. Therefor we are going to use the same number of layers as the base model. Neural Network model used is,

```
model = Sequential()
model.add(Dense(120, activation='relu', input_shape = (n_cols,)))
model.add(Dense(120, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
```
![roc_nn](https://user-images.githubusercontent.com/24527000/56231853-6e40ce80-604d-11e9-9717-70184e6afe57.png)

Neural Network gives a roc_auc value of 0.0.87594. This is slightly less than both LGBMClassifier and RandomForestClassifier.

## Conclusion

The best classifier to predict if it is going to rain tomorrow given weather data set provided is the RandomForestClassifer with parameters {'bootstrap': True, 'max_depth': 60, 'n_estimators': 300}. It gives the best roc_auc value which provides the highest probability in identifying positive cases (rain tomorrow)

```
rf_best_model = RandomForestClassifier(bootstrap = True, max_depth = 60, n_estimators = 300, random_state=123)

```
![roc_rf](https://user-images.githubusercontent.com/24527000/56231180-d1316600-604b-11e9-8589-62eeb3522ef2.png)
