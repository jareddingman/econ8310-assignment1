import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing as ES
from statsmodels.tsa.api import SimpleExpSmoothing
import numpy as np

df = pd.read_csv("https://github.com/jareddingman/econ8310-assignment1/raw/refs/heads/main/assignment_data_train.csv")
df.head()

#"The simplest model often works the best - Dusty"

#------------Exponential Smoothing-------------

dfRides = df['trips']

model = ES(dfRides)

modelFit = ES(dfRides,  
            seasonal='add', 
            seasonal_periods=24,
            ).fit(
                optimized=True)


pred = modelFit.forecast(744)
