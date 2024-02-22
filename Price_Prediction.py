import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

data.dropna(inplace=True)

from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis=1)
Y = data['median_house_value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

train_data = X_train.join(Y_train)

train_data.hist(figsize=(15,8))

train_data.ocean_proximity.value_counts()

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

train_data.corr()

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["households"] = np.log(train_data["households"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)

train_data.hist(figsize=(15,8))

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")


plt.figure(figsize=(15,8))
sns.scatterplot(data=train_data, x="longitude", y="latitude", hue="median_house_value", palette="coolwarm")

train_data['Bedroom_Ratio'] = train_data['total_rooms'] / train_data["total_bedrooms"]
train_data["house_hold_rooms"] = train_data["total_rooms"] / train_data["households"]

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, Y_train = train_data.drop(['median_house_value'], axis=1), train_data["median_house_value"]
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()

reg.fit(X_train, Y_train)

test_data = X_test.join(Y_test)

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['Bedroom_Ratio'] = test_data['total_rooms'] / test_data["total_bedrooms"]
test_data["house_hold_rooms"] = test_data["total_rooms"] / test_data["households"]

X_test, Y_test = test_data.drop(['median_house_value'], axis=1), test_data["median_house_value"]

X_test_s = scaler.transform(X_test)

reg.score(X_test,Y_test)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train_s, Y_train)

forest.score(X_test_s,Y_test)

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [30,50,100],
    "max_features": [8,12,20],
    "min_samples_split": [2,4,6,8]
}

grid_Search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

grid_Search.fit(X_train_s, Y_train)

best_forest = grid_Search.best_estimator_

best_forest.score(X_test_s,Y_test)

