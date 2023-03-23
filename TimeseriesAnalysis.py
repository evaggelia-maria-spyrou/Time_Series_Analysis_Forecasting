import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
from math import sqrt
warnings.filterwarnings("ignore")


file=pd.read_csv("timeseries.txt", delimiter = "\t", parse_dates=True, index_col=0,header=0)
df=file[:'2020']

df=df.asfreq("AS")

df = df.shift(periods = 1, freq= 'A')

df.isna().sum()

df=df.fillna(method='ffill')
df.isna().sum()

df.plot(title="Publications by Year",color='navy')
plt.show() 





def get_stationarity(data):
    adfullerTest = adfuller(data)
    print('ADF Statistic: %f' % adfullerTest[0])
    print('p-value: %f' % adfullerTest[1])
    print('Critical Values:')
    for key, value in adfullerTest[4].items():
        print('\t%s: %.3f' % (key, value))





print(get_stationarity(df.publications))





decomposition = seasonal_decompose(df.publications)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df.publications, label='publications')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()





log_df = np.log(df.publications)
log_df.plot(title="Logarithm of Publications by year",xlabel="Years")
plt.show()





print(get_stationarity(log_df))





log_df_sift =log_df- log_df.shift(1)

log_df_sift.dropna(inplace=True)
print (get_stationarity(log_df_sift))
log_df_sift.plot(title="Differencing of Publications by year",xlabel="Years")
plt.show()


# ACF




sgt.plot_acf (log_df_sift, lags = 25, color = 'g', title = ' Autocorrelation function (ACF)' ) 
plt.show () #q


# PACF




sgt.plot_pacf(log_df_sift,lags=25,method=('ols'),color = 'g', title = ' Partial Autocorrelation function (PACF)')

plt.show()
#p


# ARIMA




# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train, test = X[:'2015'], X['2016':'2019']
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error





# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))





p_values = [0, 1, 2,3,4,5]
d_values = range(0, 2)
q_values = range(0, 2)
evaluate_models(log_df, p_values, d_values, q_values)




train1=log_df[:'2015']
test1 =log_df['2016':'2019']
history = [x for x in train1]
predictions = list()
# walk-forward validation
for t in range(len(test1)):
    model1 = ARIMA(history, order=(0,1,0))
    model_fit1 = model1.fit()
    output = model_fit1.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test1[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
MAE1=round(mean_absolute_error(test1, predictions),5)
MSE1 = round(mean_squared_error(test1, predictions),3)





fig = plt.figure()
fig.suptitle("Best ARIMA(0,1,0)\n " + 'MSE='+ str(MSE1) +' MAE=' +str(MAE1))
past,= plt.plot(train1.index, train1, 'b.-', label='Train')
future ,= plt.plot(test1.index, test1, 'r.-', label='Test')
predicted_future,= plt.plot(test1.index,predictions, 'g.-', label='Predicted')
plt.legend(handles=[past, future, predicted_future])
plt.show()





train2=log_df[:'2015']
test2 =log_df['2016':'2019']
history = [x for x in train2]
predictions = list()
# walk-forward validation
for t in range(len(test2)):
    model2 = ARIMA(history, order=(1, 1, 1))
    model_fit2 = model2.fit()
    output = model_fit2.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test2[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
MAE2=round(mean_absolute_error(test2, predictions),5)


MSE2 = round(mean_squared_error(test2, predictions),3)





fig = plt.figure()
fig.suptitle("ARIMA(1,1,1)\n" +'MSE='+ str(MSE2) +' MAE=' +str(MAE2))
past,= plt.plot(train2.index, train2, 'b.-', label='Train')
future ,= plt.plot(test2.index, test2, 'r.-', label='Test')
predicted_future,= plt.plot(test2.index,predictions, 'g.-', label='Predicted')
plt.legend(handles=[past, future, predicted_future])
plt.show()


#forecast

model = ARIMA(log_df, order=(0,1,0))
results_ARIMA = model.fit()
results_ARIMA.plot_predict(1,74);
plt.title('Prediction the logarithm of publications for the next 10 years')
plt.show()

model = ARIMA(df.publications, order=(0,1,0))
results_ARIMA = model.fit()
results_ARIMA.plot_predict(1,74);
plt.title('Prediction of publications for the next 10 years')
plt.show()


# HOLT-WINTERS
log_df.plot(color='navy')
holt_winters =HWES(log_df, trend = 'add').fit().fittedvalues
holt_winters.plot(label="Holt Winters",color='cornflowerblue')
plt.title('Holt Winters Double Expontential Smoothing and Logarithm of Publications by Year',fontsize=9)
plt.legend()
plt.show()


train3=log_df[:'2015']
test3=log_df['2016':'2019']

fitted_model = HWES(train3,trend='add').fit()
test_predictions = fitted_model.forecast(4)
train3.plot(legend=True,label='train',color='cornflowerblue')
test3.plot(legend=True,label='test',figsize=(6,4),color='red')
test_predictions.plot(legend=True,label='prediction',color='green')
plt.title('Train, Test and Predicted Test using Holt Winters')

print('test:\n' + str(test3))

print('predicted:\n' + str(test_predictions))

MSE_holt=mean_squared_error(test3,test_predictions)
print('MSE: ' + str(MSE_holt))
MAE_holt=mean_absolute_error(test3,test_predictions)
print("MAE:" + str(MAE_holt))







