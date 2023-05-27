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


def modele2():
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
    
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)

    y_test = test_well['ChangeLayer']
    well_features = test_well.drop(['Facies','Formation','Well Name','Depth','NM_M','RELPOS','ChangeLayer']+old_features+['mean_GR','mean_ILD_log10','mean_DeltaPHI','mean_PHIND','mean_PE'],axis=1)
    X_test = scaler.transform(well_features)
    y_pred = clf.predict(X_test)
    test_well['Prediction'] = y_pred
    from sklearn.metrics import classification_report
    target_names = ['0','1']

    print(classification_report(y_test, y_pred,target_names=target_names))
    return test_well
