# predicting rainfall

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
Model tuning is commented out as it takes long time to run. Uncomment and run if tuning is necessary
