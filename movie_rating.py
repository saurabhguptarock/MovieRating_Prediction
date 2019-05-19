import nltk
import numpy as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense
