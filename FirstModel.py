import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from functions import *

dataset = pd.read_csv("data/world_data_really_tiny.csv")

dataset.head()