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


def modele2(tree = True):
    data = pd.read_csv("training_data.csv")
    data['ChangeLayer'] = data['Facies']

    prec_val = data['Facies'][0]
    prec_well = data['Well Name'][0]
    for i in data.index:
        prec_val_current = data['Facies'][i]
        new_Well = data['Well Name'][i]
        if(new_Well==prec_well):
            if (prec_val!=prec_val_current):
                data['ChangeLayer'][i] = 1
                prec_val = prec_val_current
                prec_well = new_Well
            else:
                data['ChangeLayer'][i] = 0
                prec_well = new_Well
        else:
            prec_val = data['Facies'][i]
            prec_well = data['Well Name'][i]
            data['ChangeLayer'][i] = 0

    TOADD = []
    old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
    def lissage(attribut):
        data["mean_"+attribut] = data[attribut].rolling(window = 4).mean()
        data["mean_"+attribut][0] = data[attribut][3]
        data["mean_"+attribut][1] = data[attribut][3]
        data["mean_"+attribut][2] = data[attribut][3]

    def derivate(attribut):
        data["der_"+attribut] = data[attribut].diff()
        data["der_"+attribut][0] = data["der_" + attribut][1]

    def meander(attribut):
        lissage(attribut)
        data["der_"+attribut] = data["mean_"+attribut].diff()
        data["der_"+attribut][0] = data["der_" + attribut][1]
        

    def new_features(method):
        global TOADD
        if method == "lissage":
            features = old_features.copy() + ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
            TOADD = ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
        if method == 'derivate':
            features = ['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
            TOADD=['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
        if method == 'meander':
            features = ['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
            TOADD=['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
        return features

    old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
    features = new_features("derivate")

    def traitement(fun_trait):
        for feature in old_features:
            fun_trait(feature)

    traitement(meander)
        
        
    test_well = data[data['Well Name'] == 'SHANKLE']
    data = data[data['Well Name'] != 'SHANKLE']

    feature_vectors = data[features]
    facies_labels = data['ChangeLayer']

    scaler = StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)

    X_train, X_cv, y_train, y_cv = train_test_split(scaled_features, facies_labels,test_size=0.05, random_state=42)
    
    if tree:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
    else:
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
        clf.add(keras.layers.Dense(units=16, activation='relu'))
        clf.add(keras.layers.Dropout(0.2))
        clf.add(keras.layers.Dense(units=16, activation='relu'))
        clf.add(keras.layers.Dropout(0.2))
        clf.add(keras.layers.Dense(units=16, activation='relu'))
        clf.add(keras.layers.Dropout(0.2))
        clf.add(keras.layers.Dense(units=16, activation='relu'))
        clf.add(keras.layers.Dropout(0.2))
        clf.add(keras.layers.Dense(units=2, activation='softmax'))
        

        clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        X_train = (X_train)/X_train.max()
        y_train = keras.utils.to_categorical(y_train, num_classes=2)

        clf.fit(X_train, y_train, epochs=500, batch_size=750, class_weight = {0:1,1:5});

    if tree:
        y_test = test_well['ChangeLayer']
        well_features = test_well.drop(['Facies','Formation','Well Name','Depth','NM_M','RELPOS','ChangeLayer']+old_features+['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE'],axis=1)
        X_test = scaler.transform(well_features)
        y_pred = clf.predict(X_test)
        test_well['Prediction'] = y_pred
        from sklearn.metrics import classification_report
        target_names = ['0','1']

        print(classification_report(y_test, y_pred,target_names=target_names))
        return test_well
    else:
        y_test = test_well['ChangeLayer']
        well_features = test_well.drop(['Facies','Formation','Well Name','Depth','NM_M','RELPOS','ChangeLayer']+old_features+['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE'],axis=1)
        X_test = scaler.transform(well_features)
        X_test = X_test/X_test.max()
        y_pred_old = clf.predict(X_test)
        y_pred = np.argmax(y_pred_old,axis=-1)
        test_well['Prediction'] = y_pred
        from sklearn.metrics import classification_report
        target_names = ['0','1']

        print(classification_report(y_test, y_pred,target_names=target_names))
        return test_well
        
        
#modele2(tree =False)

"""data = pd.read_csv("training_data.csv")
data['ChangeLayer'] = data['Facies']

prec_val = data['Facies'][0]
prec_well = data['Well Name'][0]
for i in data.index:
    prec_val_current = data['Facies'][i]
    new_Well = data['Well Name'][i]
    if(new_Well==prec_well):
        if (prec_val!=prec_val_current):
            data['ChangeLayer'][i] = 1
            prec_val = prec_val_current
            prec_well = new_Well
        else:
            data['ChangeLayer'][i] = 0
            prec_well = new_Well
    else:
        prec_val = data['Facies'][i]
        prec_well = data['Well Name'][i]
        data['ChangeLayer'][i] = 0

TOADD = []
old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
def lissage(attribut):
    data["mean_"+attribut] = data[attribut].rolling(window = 4).mean()
    data["mean_"+attribut][0] = data[attribut][3]
    data["mean_"+attribut][1] = data[attribut][3]
    data["mean_"+attribut][2] = data[attribut][3]

def derivate(attribut):
    data["der_"+attribut] = data[attribut].diff()
    data["der_"+attribut][0] = data["der_" + attribut][1]

def meander(attribut):
    lissage(attribut)
    data["der_"+attribut] = data["mean_"+attribut].diff()
    data["der_"+attribut][0] = data["der_" + attribut][1]
    

def new_features(method):
    global TOADD
    if method == "lissage":
        features = old_features.copy() + ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
        TOADD = ['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE']
    if method == 'derivate':
        features = ['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
        TOADD=['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
    if method == 'meander':
        features = ['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
        TOADD=['der_GR','der_ILD_log10','der_DeltaPHI','der_PHIND','der_PE']
    if method == 'default':
        features = old_features.copy()
    return features

old_features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND','PE']
features = new_features("derivate")

def traitement(fun_trait):
    for feature in old_features:
        fun_trait(feature)

traitement(meander)
    
    
test_well = data[data['Well Name'] == 'SHANKLE']
data = data[data['Well Name'] != 'SHANKLE']

feature_vectors = data[features]
facies_labels = data['ChangeLayer']

scaler = StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, facies_labels,test_size=0.2, random_state=42)



#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier()
#clf.fit(X_train,y_train)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def weighted_binary_crossentropy(y_true, y_pred):
    weights = K.sum(y_true, axis=0) / K.sum(y_true)
    weights = K.cast(weights, K.floatx())
    loss = K.binary_crossentropy(y_true, y_pred)
    return loss * weights

clf = keras.Sequential()

clf.add(keras.layers.Dense(units=64, activation='relu', input_dim=5))
clf.add(keras.layers.Dropout(0.5))
clf.add(keras.layers.Dense(units=32, activation='relu'))
clf.add(keras.layers.Dropout(0.5))
clf.add(keras.layers.Dense(units=16, activation='relu'))
clf.add(keras.layers.Dropout(0.2))
clf.add(keras.layers.Dense(units=2, activation='softmax'))

clf.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy'])

X_train = (X_train)/X_train.max()
y_train = keras.utils.to_categorical(y_train, num_classes=2)


clf.fit(X_train, y_train, epochs=3000, batch_size=650);





y_test = test_well['ChangeLayer']
well_features = test_well.drop(['Facies','Formation','Well Name','Depth','NM_M','RELPOS','ChangeLayer']+old_features+['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE'],axis=1)
X_test = scaler.transform(well_features)
X_test = X_test/X_test.max()
y_pred_old= clf.predict(X_test)
y_pred = np.argmax(y_pred_old,axis=-1)
test_well['Prediction'] = y_pred
from sklearn.metrics import classification_report
target_names = ['0','1']

print(classification_report(y_test, y_pred,target_names=target_names))

def compare_facies_plot_der(logs):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            ["#FFFFFF","#000000"], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['ChangeLayer'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs['Prediction'].values,1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))
    ax[0].plot(logs.der_GR, logs.Depth, '-g')
    ax[1].plot(logs.der_ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.der_DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.der_PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.der_PE, logs.Depth, '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=1)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=0,vmax=1)

    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("d.GR")
    ax[0].set_xlim(logs.der_GR.min(),logs.der_GR.max())
    ax[1].set_xlabel("d.ILD")
    ax[1].set_xlim(logs.der_ILD_log10.min(),logs.der_ILD_log10.max())
    ax[2].set_xlabel("d.DeltaPHI")
    ax[2].set_xlim(logs.der_DeltaPHI.min(),logs.der_DeltaPHI.max())
    ax[3].set_xlabel("d.PHIND")
    ax[3].set_xlim(logs.der_PHIND.min(),logs.der_PHIND.max())
    ax[4].set_xlabel("d.PE")
    ax[4].set_xlim(logs.der_PE.min(),logs.der_PE.max())
    ax[5].set_xlabel('Facies')
    ax[6].set_xlabel("Prediction")

    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([]); ax[6].set_xticklabels([]); ax[5].set_yticklabels([])


    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)

compare_facies_plot_der(test_well)
plt.show()"""