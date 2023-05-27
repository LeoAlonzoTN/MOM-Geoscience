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


def modele1():
    """This model is a simple model to predict each layer. It uses NeuralNetwork from TensorFlow"""

    data = pd.read_csv("training_data.csv")
    TOADD = []
    old_features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']

    def lissage(attribut):
        data["mean_"+attribut] = data[attribut].rolling(window=4).mean()
        data["mean_"+attribut][0] = data[attribut][3]
        data["mean_"+attribut][1] = data[attribut][3]
        data["mean_"+attribut][2] = data[attribut][3]

    def derivate(attribut):
        data["der_"+attribut] = data[attribut].diff()
        data["der_"+attribut][0] = data["der_" + attribut][1]

    def new_features(method):
        global TOADD
        if method == "lissage":
            features = old_features.copy(
            ) + ['mean_GR', 'mean_ILD_log10', 'mean_DeltaPHI', 'mean_PHIND', 'mean_PE']
            TOADD = ['mean_GR', 'mean_ILD_log10',
                     'mean_DeltaPHI', 'mean_PHIND', 'mean_PE']
        if method == 'derivate':
            features = old_features.copy(
            ) + ['der_GR', 'der_ILD_log10', 'der_DeltaPHI', 'der_PHIND', 'der_PE']
            TOADD = ['der_GR', 'der_ILD_log10',
                     'der_DeltaPHI', 'der_PHIND', 'der_PE']
        return features

    old_features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    features = old_features.copy()

    test_well = data[data['Well Name'] == 'SHANKLE']
    data = data[data['Well Name'] != 'SHANKLE']

    feature_vectors = data[features]
    facies_labels = data['Facies']

    scaler = StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)

    X_train, X_cv, y_train, y_cv = train_test_split(
        scaled_features, facies_labels, test_size=0.05, random_state=42)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    clf = keras.Sequential()

    clf.add(keras.layers.Dense(units=64, activation='relu', input_dim=5))
    clf.add(keras.layers.Dropout(0.2))
    clf.add(keras.layers.Dense(units=32, activation='relu'))
    clf.add(keras.layers.Dropout(0.2))
    clf.add(keras.layers.Dense(units=16, activation='relu'))
    clf.add(keras.layers.Dropout(0.2))
    clf.add(keras.layers.Dense(units=9, activation='softmax'))

    clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = (X_train)/X_train.max()
    y_train = keras.utils.to_categorical(y_train-1, num_classes=9)

    clf.fit(X_train, y_train, epochs=80, batch_size=300);


    y_test = test_well['Facies']
    well_features = test_well.drop(['Facies', 'Formation', 'Well Name', 'Depth', 'NM_M', 'RELPOS'], axis=1)
    X_test = scaler.transform(well_features)
    X_test = X_test/X_test.max()
    y_pred_old = clf.predict(X_test)
    y_pred = np.argmax(y_pred_old,axis=-1)+1
    test_well['Prediction'] = y_pred
    from sklearn.metrics import classification_report, accuracy_score
    try: # in case our classification didn't predict any BS (it can happen)
        target_names = ['SS', 'CSiS', 'FSiS', 'SiSh','MS', 'WS', 'D','PS','BS']
        print("modele1:")
        cr = classification_report(y_test, y_pred,target_names=target_names)
        print(cr)
        print(accuracy_score(y_test,y_pred))
    except:
        target_names = ['SS', 'CSiS', 'FSiS', 'SiSh','MS', 'WS', 'D','PS']
        print("modele1:")
        cr = classification_report(y_test, y_pred,target_names=target_names)
        print(cr)
        print(accuracy_score(y_test,y_pred))
    return (test_well,accuracy_score(y_test,y_pred))