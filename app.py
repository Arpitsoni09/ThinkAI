import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix , classification_report


model = load_model('C:\\SPPAPP\\Stock_Predictions_Model.keras')

st.header('Stock Price Predictor with graph' )

stock =st.text_input('Enter Stock Symbol', "GOOG" )
start = '2015-01-22'
end = date.today()

data = yf.download(stock, start ,end)
data.info()
data.head()



st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))

plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

text_input = stock

st.header('Prediction Value' )

stock =st.text_input('Enter Stock', text_input)
data = yf.download(stock, start ,end)





start = '2015-01-01'
end = date.today()
stock = 'GOOG'

data = yf.download(stock, start, end)










# ... (existing code up to prediction)

# Calculate predicted high and low (replace with your logic)
high_pred = np.max(predict)  # Assuming model predicts actual prices
# OR (replace with your alternative calculation)
# high_threshold = current_price * 1.05  # 5% increase from current price
# high_pred = predict[0][0] * scale + high_threshold  # Example using threshold

low_pred = np.min(predict)  # Assuming model predicts actual prices
# OR (replace with your alternative calculation)
# low_threshold = current_price * 0.95  # 5% decrease from current price
# low_pred = predict[0][0] * scale + low_threshold  # Example using threshold

# Display predicted high and low values
st.header('High and Low Prediction')
st.write(f'Predicted High: {high_pred}')
st.write(f'Predicted Low: {low_pred}')

# ... (rest of the code)


st.subheader('Original Predicted Price')
fig5 = plt.figure(figsize=(8,6))

plt.plot(y, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig5)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Convert predictions to binary labels
predicted_labels = np.where(predict[1:] > predict[:-1], 1, 0)
actual_value = np.where(y[1:] > y[:-1], 1, 0)

# Generate confusion matrix
conf_matrix = confusion_matrix(actual_value, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['True 0', 'True 1'])


for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="Green" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.pyplot()
