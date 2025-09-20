# %% [markdown]
# # Natural Language Processing: Emotions Sentiment Analysis using TensorFlow with LSTM

# %% [markdown]
# ## Install Kaggle and Config Kaggle

# %%
# Install Kaggle using pip
!pip install -q kaggle

# %%
from google.colab import files
files.upload()

# %%
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

# %% [markdown]
# ## Download Dataset from Kaggle

# %%
# Download Dataset
!kaggle datasets download -d abdallahwagih/emotion-dataset

# %% [markdown]
# ## Import Required Libraries

# %%
# Import Required Libraries
import zipfile
import numpy as np
import pandas as pd

import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout

# %% [markdown]
# ## Extract File Zip

# %%
# Extract File Zip
data_dir = "/content/"
zip_dir = f"{data_dir}/emotion-dataset.zip"
zip = zipfile.ZipFile(zip_dir, 'r')
zip.extractall('/content')
zip.close()

# %% [markdown]
# ## Open File Dataset

# %%
# Open Dataset File
data = pd.read_csv(f"{data_dir}Emotion_classify_Data.csv")
data.head()

# %% [markdown]
# ## Search Total Data, Total Null Values, and Total Duplicated Values

# %%
# Search Total Data, Total Null Values, and Total Duplicated Values
total_data = data.shape
null_data = data.isnull().sum()
duplicated_data  = data.duplicated()
print(f"Total Row & Column: \n{total_data}\n\nTotal null Values:\n{null_data}\n\nTotal duplicated Values:\n{duplicated_data}")

# %% [markdown]
# ## Count Emotions Value

# %%
data['Emotion'].value_counts()

# %% [markdown]
# ## Countplot Emotions

# %%
# Countplot Emotion
sb.countplot(data.Emotion)

# %% [markdown]
# ## Import Another Required Library

# %%
# Import Required Library
import nltk
import keras
import re, string, unicodedata

from string import punctuation
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from nltk import pos_tag
from nltk.corpus import wordnet

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

nltk.download('stopwords')

# %% [markdown]
# ## Stopwords

# %%
stwd = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stwd.update(punctuation)

# %% [markdown]
# ## Cleaning Text

# %%
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stwd = stopwords.words('english')

def clean_text(text):

    cleaned_text = text
    cleaned_text = strip_html(cleaned_text)
    cleaned_text = remove_between_square_brackets(cleaned_text)
    cleaned_text = remove_url(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)

    return cleaned_text

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_url(text):
    return re.sub(r'http\S+', '', text)

def remove_stopwords(text):
    final_text = []
    for word in text.split():
        if word.strip().lower() not in stwd:
            final_text.append(word.strip())
    return " ".join(final_text)

data['Comment'] = data['Comment'].apply(clean_text)

# %%
def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(data.Comment)
corpus[:10]

# %%
# Total Common Words
from collections import Counter
counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = dict(most_common)
most_common

# %%
from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    cv = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_words = cv.transform(corpus)
    sum_words = bag_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# %% [markdown]
# ## Most Common Words Countplot

# %%
plt.figure(figsize = (16,9))
most_common = get_top_text_ngrams(data.Comment, 10, 1)
most_common = dict(most_common)
sb.barplot(x=list(most_common.values()),y=list(most_common.keys()))

# %% [markdown]
# ## One-Hot Encoding

# %%
# Data Emotion One-Hot Encoding
emotion = pd.get_dummies(data.Emotion, dtype=int)
new_data = pd.concat([data, emotion], axis=1)
new_data = new_data.drop(columns='Emotion')
new_data.head(10)

# %%
new_data.columns

# %%
datas = new_data['Comment'].values
emotions = new_data[['anger', 'fear', 'joy']].values

# %%
datas

# %%
emotions

# %% [markdown]
# ## Tokenizer and Train Model

# %%
x_train,x_test,y_train,y_test = train_test_split(datas, emotions, test_size = 0.2, shuffle=True)

# %%
# Create Callback
class Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.98 and logs.get('val_accuracy')> 0.92):
      print("Accuracy have been achieved!")
      self.model.stop_training = True
callbacks= Callback()

# %%
vocab_size = 5000
max_len = 200
trunc_type = "post"
oov_tok = "<oov>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index

sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)
pad_train = pad_sequences(sequences_train, maxlen=max_len, truncating=trunc_type)
pad_test = pad_sequences(sequences_test, maxlen=max_len, truncating=trunc_type)

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy',)
model.summary()

# %%
history = model.fit(pad_train, y_train, epochs=50, validation_data=(pad_test, y_test), batch_size=64, verbose=2, callbacks=[callbacks])

# %% [markdown]
# ## Modela Evaluation: Model Accuracy and Model Loss Plot

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Create a subplot figure
plt.figure(figsize=(20, 5))  # Adjust figure size as needed

# Subplot for Accuracy
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.plot(epochs, acc, 'r', label='Train Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend(loc=0)

# Subplot for all
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.plot(epochs, acc, 'blue', label='Train Accuracy')
plt.plot(epochs, val_acc, 'green', label='Validation Accuracy')
plt.plot(epochs, loss, 'red', label='Train Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.title('Model Accuracy and Loss')
plt.legend(loc=0)


# Adjust layout to prevent overlapping elements
plt.tight_layout()

plt.show()


