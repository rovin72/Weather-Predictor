import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
weather = pd.read_csv('weatherstats_ottawa_forecast_hourly.csv')
weather = weather.dropna(axis=1, how='all')
y = weather.temperature
# removes outliers
y = np.clip(y, y.quantile(0.05), y.quantile(0.95))  # Limit extreme values
independant=['date_time_local','pop','windchill','wind_speed','humidex']
x= weather[independant]
# Convert all non-numeric columns to proper numeric format
for col in x.columns:
    if x[col].dtype == 'object':  # If a column is still a string
        x[col] = pd.to_numeric(x[col], errors='coerce') 
# Fill missing values in numerical columns with their mean
x = x.fillna(0)
x['date_time_local'] = pd.to_datetime(x['date_time_local'], errors='coerce')
x['hour'] = x['date_time_local'].dt.hour
x['day'] = x['date_time_local'].dt.day
x['month'] = x['date_time_local'].dt.month
x['weekday'] = x['date_time_local'].dt.weekday
x = x.drop(columns=['date_time_local'])
weather_tree=DecisionTreeRegressor()
weather_linear= LinearRegression()
weather_forest=RandomForestRegressor()
weather_XGB_MSE_R2=xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
linearMSE=cross_val_score(weather_linear, x, y, cv=10, scoring='neg_mean_squared_error')
linearMAE=cross_val_score(weather_linear, x, y, cv=10, scoring='neg_mean_absolute_error')
linearR2=cross_val_score(weather_linear, x, y, cv=10,scoring='r2')
treeMSE=cross_val_score(weather_tree, x, y, cv=10, scoring='neg_mean_squared_error')
treeMAE=cross_val_score(weather_tree, x, y, cv=10, scoring='neg_mean_absolute_error')
treeR2=cross_val_score(weather_tree, x, y, cv=10, scoring='r2')
forestMSE=cross_val_score(weather_forest, x, y, cv=10, scoring='neg_mean_squared_error')
forestMAE=cross_val_score(weather_forest, x, y, cv=10, scoring='neg_mean_absolute_error')
forestR2=cross_val_score(weather_forest, x, y, cv=10, scoring='r2')
XGBmse=cross_val_score(weather_XGB_MSE_R2,x,y,cv=10,scoring='neg_mean_squared_error')
XGBmae=cross_val_score(weather_XGB_MSE_R2,x,y,cv=10,scoring='neg_mean_absolute_error')
XGBr2=cross_val_score(weather_XGB_MSE_R2,x,y,cv=10,scoring='r2')
linearRMSE=[np.sqrt(i) for i in linearMSE]
treeRMSE=[np.sqrt(i) for i in treeMSE]
linearRMSE=[np.sqrt(i) for i in linearMSE]
treeRMSE=[np.sqrt(i) for i in treeMSE]
figure,axis=plt.subplots(2,2)
modeltypes=['Linear Regression','Descision Tree','Random Forest', 'XGBoost']
MSEvalues=[np.sqrt(-linearMSE.mean()),np.sqrt(-treeMSE.mean()),np.sqrt(-forestMSE.mean()),np.sqrt(-XGBmse.mean())]
MSEconfidenceIntervalsRange=[1.96*(np.std([np.sqrt(i)for i in linearMSE])/np.sqrt(len(linearMSE))),1.96*(np.std([np.sqrt(i)for i in treeMSE])/np.sqrt(len(treeMSE))),1.96*(np.std([np.sqrt(i)for i in forestMSE])/np.sqrt(len(forestMSE))),1.96*(np.std([np.sqrt(i)for i in XGBmse])/np.sqrt(len(XGBmse)))]
MAEvalues=[-linearMAE.mean(),-treeMAE.mean(),-forestMAE.mean(),-XGBmae.mean()]
MAEconfidenceIntervalsRange=[1.96*(np.std(linearMAE)/np.sqrt(len(linearMAE))),1.96*(np.std(treeMAE)/np.sqrt(len(treeMAE))),1.96*(np.std(forestMAE)/np.sqrt(len(forestMAE))),1.96*(np.std(XGBmae)/np.sqrt(len(XGBmae)))]
R2values=[linearR2.mean(),treeR2.mean(),forestR2.mean(),XGBr2.mean()]
R2confidenceIntervalsRange=[1.96*(np.std(linearR2)/np.sqrt(len(linearR2))),1.96*(np.std(treeR2)/np.sqrt(len(treeR2))),1.96*(np.std(forestR2)/np.sqrt(len(forestR2))),1.96*(np.std(XGBr2)/np.sqrt(len(XGBr2)))]
axis[0,0].bar(modeltypes,MSEvalues,yerr=MSEconfidenceIntervalsRange)
axis[0,0].set_title('MSE comparision')
axis[0,0].set_xlabel('Model Types')
axis[0,0].set_ylabel('MSE')
#axis[0,0].set_yscale('log')
axis[0,1].bar(modeltypes,MAEvalues,yerr=MAEconfidenceIntervalsRange)
axis[0,1].set_title('MAE comparision')
axis[0,1].set_xlabel('Model Types')
axis[0,1].set_ylabel('MAE')
#axis[0,1].set_yscale('log')
axis[1,0].bar(modeltypes,R2values,yerr=R2confidenceIntervalsRange)
axis[1,0].set_title('R^2 comparision')
axis[1,0].set_xlabel('Model Types')
axis[1,0].set_ylabel('R^2')
plt.tight_layout()
plt.show()
print(-linearMSE.mean())
print(-linearMAE.mean())
print(-treeMSE.mean())
print(-treeMAE.mean())
print(-forestMSE.mean())  
print(-forestMAE.mean())  
print(-XGBmse.mean())
print(-XGBmae.mean())