import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import optuna
from tqdm import tqdm
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from optuna.integration import LightGBMPruningCallback

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score, classification_report, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc