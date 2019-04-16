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

Furthur explorations shows that there are no outliers in the data and all the data points lie within acceptable ranges.
The correlation matrix for the feature varibles and target shown here.

https://github.com/dinushawiki/predicting_rainfall/master/correlation_matrix.png?raw=true


