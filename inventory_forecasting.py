import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime
import holidays

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("StoreDemand.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Feature engineering
parts = df["date"].str.split("-", expand=True)

df["year"] = parts[0].astype(int)
df["month"] = parts[1].astype(int)
df["day"] = parts[2].astype(int)

# Weekend feature
def weekend_or_weekday(year, month, day):
    d = datetime(year, month, day)
    return 1 if d.weekday() >= 5 else 0

df["weekend"] = df.apply(lambda x: weekend_or_weekday(x["year"], x["month"], x["day"]), axis=1)

# Weekday feature
df["weekday"] = df.apply(lambda x: datetime(x["year"], x["month"], x["day"]).weekday(), axis=1)

# Holiday feature
india_holidays = holidays.country_holidays('IN')
df["holidays"] = df["date"].apply(lambda x: 1 if x in india_holidays else 0)

# Cyclical month feature
df["m1"] = np.sin(df["month"] * (2 * np.pi / 12))
df["m2"] = np.cos(df["month"] * (2 * np.pi / 12))

# Drop date column
df.drop("date", axis=1, inplace=True)

# EDA
plt.figure(figsize=(6,4))
sb.histplot(df["sales"])
plt.title("Sales Distribution")
plt.show()

# Remove outliers
df = df[df["sales"] < 140]

# Features and target
X = df.drop(["sales","year"], axis=1)
y = df["sales"]

# Train test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=22
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Models
models = [
    LinearRegression(),
    XGBRegressor(),
    Lasso(),
    Ridge()
]

# Training
for model in models:

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    print(model)
    print("Training MAE:", mean_absolute_error(y_train, train_preds))
    print("Validation MAE:", mean_absolute_error(y_val, val_preds))
    print()