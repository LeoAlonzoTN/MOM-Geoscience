import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


data = pd.read_csv("training_data.csv")
data['ChangeLayer'] = data['Facies']

prec_val = data['Facies'][0]
for i in data.index:
    prec_val_current = data['Facies'][i]
    if (prec_val!=prec_val_current):
        data['ChangeLayer'][i] = 1
        prec_val = prec_val_current
    else:
        data['ChangeLayer'][i] = 0

TOADD = []
old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
def lissage(attribut):
    data["mean_"+attribut] = data[attribut].rolling(window = 4).mean()
    data["mean_"+attribut][0] = data[attribut][3]
    data["mean_"+attribut][1] = data[attribut][3]
    data["mean_"+attribut][2] = data[attribut][3]

def derivate(attribut):
    """Derivate the value of the column attribut"""
    data["der_"+attribut] = data[attribut].diff()
    data["der_"+attribut][0] = data["der_" + attribut][1]

def new_features(method):
    """Accepts method lissage and derivate"""
    global TOADD
    if method == "lissage":
        features = old_features.copy() + ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
        TOADD = ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
    if method == 'derivate':
        features = old_features.copy() + ['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
        TOADD=['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
    return features

old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
features = new_features("derivate")

def traitement(fun_trait):
    for feature in old_features:
        fun_trait(feature)

traitement(derivate)
    
    
test_well = data[data['Well Name'] == 'SHANKLE']
data = data[data['Well Name'] != 'SHANKLE']


feature_vectors = data[features]
facies_labels = data['ChangeLayer']
print(test_well[test_well['ChangeLayer']==1])

scaler = StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)