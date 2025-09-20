# %% [markdown]
# # Tesla Stock Prediction using LSTM Deep Learning Model
# 

# %% [markdown]
# ### Install & Config Kaggle

# %% [markdown]
# #### Install Kaggle

# %%
!pip install -q kaggle

# %% [markdown]
# #### Connect to Google Drive

# %%
from google.colab import files
files.upload()

# %% [markdown]
# #### Config Kaggle

# %%
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

# %%
!kaggle datasets download -d varpit94/tesla-stock-data-updated-till-28jun2021

# %% [markdown]
# ## Import Required Libraries

# %%
import sklearn
import zipfile
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### Extract Dataset

# %%
# Extract file zip
data_dir = "/content/"
zip_dir = f"{data_dir}tesla-stock-data-updated-till-28jun2021.zip"
zip = zipfile.ZipFile(zip_dir, 'r')
zip.extractall('/content')
zip.close()

# %% [markdown]
# ## Load Dataset

# %%
data = pd.read_csv(f"{data_dir}TSLA.csv")
data.head(len(data))

# %% [markdown]
# ## Data Assesing

# %%
data.shape

# %%
data.isnull().sum()

# %%
data.isna().sum()

# %%
data = data.drop(columns=['High',	'Low',	'Close',	'Volume',	'Adj Close'], axis=1)

# %%
data.head()

# %% [markdown]
# ## Normalize Data and Create Plot

# %%
scaler = MinMaxScaler()

data['Date'] = pd.to_datetime(data['Date'])
date = data['Date'].values
data['Open'] = scaler.fit_transform(np.array(data['Open']).reshape(-1, 1))
open = data['Open'].values

plt.figure(figsize=(15, 5))
plt.plot(date, open)
plt.title('TSLA Open Stock Prices 2010 - 2022')
plt.xlabel('Years')
plt.ylabel('Open Stock Prices (Normalize)')
plt.show()

# %%
data.head(len(data))

# %% [markdown]
# ## Setting Threshold MAE Value

# %%
threshold_mae = (data['Open'].max() - data['Open'].min()) * 10/100
print(f"MAX: {data['Open'].max():.16f}\nMIN: {data['Open'].min():.16f}\nThreshold MAE: {threshold_mae:.16f}")

# %% [markdown]
# ## Create Callback

# %%
# Create Callback
class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < threshold_mae):
      print("\nThe MAE is lower than the Threshold MAE!")
      self.model.stop_training = True
callbacks= Callback()

# %% [markdown]
# ## Data Splitting

# %%
x_train, x_test, y_train, y_test = train_test_split(open, date, test_size = 0.2, shuffle=False)

# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  data_series = tf.data.Dataset.from_tensor_slices(series)
  data_series = data_series.window(window_size + 1, shift=1, drop_remainder = True)
  data_series = data_series.flat_map(lambda w: w.batch(window_size + 1))
  data_series = data_series.shuffle(shuffle_buffer)
  data_series = data_series.map(lambda w: (w[:-1], w[-1:]))
  return data_series.batch(batch_size).prefetch(1)

# %%
data_training = windowed_dataset(x_train, window_size=64, batch_size=128, shuffle_buffer=1000)
data_testing = windowed_dataset(x_test, window_size=64, batch_size=128, shuffle_buffer=1000)

# %% [markdown]
# ## Build Model

# %%
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1),
])

# %% [markdown]
# ## Train Model

# %%
optimizers = tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizers,
              metrics=["mae"])

history = model.fit(
    data_training,
    epochs=50,
    validation_data=data_testing,
)

# %% [markdown]
# ## Model Evaluation
# 

# %% [markdown]
# ### Model MAE

# %%
plt.figure(figsize=(15, 5))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Mae')
plt.ylabel('Mae')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# %% [markdown]
# ### Model Loss

# %%
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show()


