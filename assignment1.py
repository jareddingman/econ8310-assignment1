import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np

df = pd.read_csv("https://github.com/jareddingman/econ8310-assignment1/raw/refs/heads/main/assignment_data_train.csv")
df.head()

#"The simplest model often works the best - Dusty"

#------------Exponential Smoothing-------------

dfRides = df['trips']
alpha020 = SimpleExpSmoothing(dfRides).fit(
                    smoothing_level=0.2,
                    optimized=False)
level2 = alpha020.forecast(1)
print(level2)

model = SimpleExpSmoothing(dfRides).fit(
                    optimized=True)

modelFit = ExponentialSmoothing(dfRides,  
            seasonal='add', 
            seasonal_periods=24,
            ).fit()

pred = modelFit.forecast(744)