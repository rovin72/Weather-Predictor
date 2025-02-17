#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#importing dataset
weather = pd.read_csv('weatherstats_ottawa_forecast_hourly.csv')
#drops empty columns
weather = weather.dropna(axis=1, how='all')
#makes temperature dependant
y = weather.temperature
#sets x to independant variables
independant=['date_time_local','pop','windchill','wind_speed','humidex','conditions', 'pop_category', 'wind_direction']
x= weather[independant]
#converts date into key hour day month and weekdays
x['date_time_local'] = pd.to_datetime(x['date_time_local'], errors='coerce')
x['hour'] = x['date_time_local'].dt.hour
x['day'] = x['date_time_local'].dt.day
x['month'] = x['date_time_local'].dt.month
x['weekday'] = x['date_time_local'].dt.weekday
x = x.drop(columns=['date_time_local'])
# handles categorical variables by one-hot encoding them
categorical_cols = ['conditions', 'pop_category', 'wind_direction']
x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)  # One-hot encoding
#replaces any missing values with mean
x=x.fillna(x.mean())
#creates models
weather_tree=DecisionTreeRegressor()
weather_linear= LinearRegression()
weather_forest=RandomForestRegressor()
weather_XGB_MSE_R2=xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
#fits models
linearMSE=cross_val_score(weather_linear, x, y, cv=5, scoring='neg_mean_squared_error')
linearMAE=cross_val_score(weather_linear, x, y, cv=5, scoring='neg_mean_absolute_error')
linearR2=cross_val_score(weather_linear, x, y, cv=5,scoring='r2')
treeMSE=cross_val_score(weather_tree, x, y, cv=5, scoring='neg_mean_squared_error')
treeMAE=cross_val_score(weather_tree, x, y, cv=5, scoring='neg_mean_absolute_error')
treeR2=cross_val_score(weather_tree, x, y, cv=5, scoring='r2')
forestMSE=cross_val_score(weather_forest, x, y, cv=5, scoring='neg_mean_squared_error')
forestMAE=cross_val_score(weather_forest, x, y, cv=5, scoring='neg_mean_absolute_error')
forestR2=cross_val_score(weather_forest, x, y, cv=5, scoring='r2')
XGBmse=cross_val_score(weather_XGB_MSE_R2,x,y,cv=5,scoring='neg_mean_squared_error')
XGBmae=cross_val_score(weather_XGB_MSE_R2,x,y,cv=5,scoring='neg_mean_absolute_error')
XGBr2=cross_val_score(weather_XGB_MSE_R2,x,y,cv=5,scoring='r2')
#converts MSE to RMSE
linearRMSE=[np.sqrt(-i) for i in linearMSE]
treeRMSE=[np.sqrt(-i) for i in treeMSE]
forestRMSE=[np.sqrt(-i) for i in forestMSE]
XGBrmse=[np.sqrt(-i) for i in XGBmse]
#preps to plots all graphs
figure,axis=plt.subplots(2,2)
modeltypes=['Linear Regression','Descision Tree','Random Forest', 'XGBoost']
#confidence interval and mean calculations
MSEvalues=[np.sqrt(-linearMSE.mean()),np.sqrt(-treeMSE.mean()),np.sqrt(-forestMSE.mean()),np.sqrt(-XGBmse.mean())]
MSEconfidenceIntervalsRange=[1.96*(np.std(linearRMSE)/np.sqrt(len(linearMSE))),1.96*(np.std(treeRMSE)/np.sqrt(len(treeMSE))),1.96*(np.std(forestRMSE)/np.sqrt(len(forestMSE))),1.96*(np.std(XGBrmse)/np.sqrt(len(XGBmse)))]
MAEvalues=[-linearMAE.mean(),-treeMAE.mean(),-forestMAE.mean(),-XGBmae.mean()]
MAEconfidenceIntervalsRange=[1.96*(np.std(linearMAE)/np.sqrt(len(linearMAE))),1.96*(np.std(treeMAE)/np.sqrt(len(treeMAE))),1.96*(np.std(forestMAE)/np.sqrt(len(forestMAE))),1.96*(np.std(XGBmae)/np.sqrt(len(XGBmae)))]
R2values=[linearR2.mean(),treeR2.mean(),forestR2.mean(),XGBr2.mean()]
R2confidenceIntervalsRange=[1.96*(np.std(linearR2)/np.sqrt(len(linearR2))),1.96*(np.std(treeR2)/np.sqrt(len(treeR2))),1.96*(np.std(forestR2)/np.sqrt(len(forestR2))),1.96*(np.std(XGBr2)/np.sqrt(len(XGBr2)))]
#plots RMSE graph
axis[0,0].bar(modeltypes,MSEvalues,yerr=MSEconfidenceIntervalsRange)
axis[0,0].set_title('RMSE comparision')
axis[0,0].set_xlabel('Model Types')
axis[0,0].set_ylabel('RMSE')
#plots MAE graph
axis[0,1].bar(modeltypes,MAEvalues,yerr=MAEconfidenceIntervalsRange)
axis[0,1].set_title('MAE comparision')
axis[0,1].set_xlabel('Model Types')
axis[0,1].set_ylabel('MAE')
#plots R2 graph
axis[1,0].bar(modeltypes,R2values,yerr=R2confidenceIntervalsRange)
axis[1,0].set_title('R^2 comparision')
axis[1,0].set_xlabel('Model Types')
axis[1,0].set_ylabel('R^2')
plt.tight_layout()
plt.show()
#creates new model
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(x, y)
#gets importance
importances = forest.feature_importances_
feature_names = x.columns

# Sort feature importance
sorted_idx = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.show()

# Trains the model based on XGB
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(x, y)

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.show()

#prints summary
print(-linearMSE.mean())
print(-linearMAE.mean())
print(linearR2.mean())
print(-treeMSE.mean())
print(-treeMAE.mean())
print(treeR2.mean())
print(-forestMSE.mean())  
print(-forestMAE.mean())  
print(forestR2.mean())
print(-XGBmse.mean())
print(-XGBmae.mean())
print(XGBr2.mean())
