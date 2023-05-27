import modele1
import modele2
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
from finalPlot import plotFinal


print("loading modele 1 ...")
test_well_pred1,accuracy = modele1.modele1()
print("loading modele 2 ...")
test_well_pred2 = modele2.modele2()
print("Done")
Modele_final = pd.DataFrame()
Modele_final[['Facies','Old_Prediction']] = test_well_pred1[['Facies','Prediction']]
Modele_final['New_Prediction'] = test_well_pred1['Prediction']
Modele_final[['Change_Layer','Change_Layer_Predicted']] = test_well_pred2[['ChangeLayer','Prediction']]
Modele_final['Entropy'] = 1
Modele_final['Depth'] = test_well_pred1['Depth']
i= test_well_pred1.index[0]


entropy = []
while i <= test_well_pred1.index[-1]:
    debut = i
    cur_dict = {}
    while test_well_pred2['Prediction'][i] != 1 and i < test_well_pred1.index[-1]:
        if test_well_pred1['Prediction'][i] in cur_dict: 
            cur_dict[test_well_pred1['Prediction'][i]] +=1
        else:
            cur_dict[test_well_pred1['Prediction'][i]] =1
        i+=1
    if test_well_pred1['Prediction'][i] in cur_dict: 
        cur_dict[test_well_pred1['Prediction'][i]] +=1
    else:
        cur_dict[test_well_pred1['Prediction'][i]] =1
    max_facies = max(cur_dict,key=cur_dict.get)
    if debut-i != 0: 
        Modele_final.loc[debut:i,'Entropy'] = cur_dict[max_facies]/sum(cur_dict.values())
    Modele_final.loc[debut:i,'New_Prediction'] = max_facies
    i+=1


# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
test_well_pred1.loc[:,'FaciesLabels'] = test_well_pred1.apply(lambda row: label_facies(row, facies_labels), axis=1)

def compare_facies_plot(logs, compadre, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[compadre].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im2, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    ax[6].set_xlabel(compadre)
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[6].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)

y_test = Modele_final['Facies']
y_pred = Modele_final['New_Prediction']
from sklearn.metrics import classification_report
try:
    target_names = ['SS','CSiS', 'FSiS', 'SiSh','MS', 'WS', 'D','PS','BS']
    print(y_test.unique())

    print(classification_report(y_test, y_pred,target_names=target_names))
    print("modele1 accuracy : ", accuracy)
except:
    target_names = ['SS','CSiS', 'FSiS', 'SiSh','MS', 'WS', 'D','PS']
    print(y_test.unique())

    print(classification_report(y_test, y_pred,target_names=target_names))
    print("modele1 accuracy : ", accuracy)

#compare_facies_plot(Modele_final, 'New_Prediction', facies_colors)
#ajouter le plot de leo
#plt.show()

plotFinal(Modele_final,facies_colors)
plt.show()


    
    