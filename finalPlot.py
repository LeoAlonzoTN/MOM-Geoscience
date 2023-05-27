import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotFinal(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    cmap_sep = colors.ListedColormap(
            ["#FFFFFF","#000000"],'indexed')

    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs['Change_Layer'].values,1), 100, 1)
    cluster3 = np.repeat(np.expand_dims(logs['Change_Layer_Predicted'].values,1), 100, 1)
    cluster4 = np.repeat(np.expand_dims(logs['Old_Prediction'].values,1), 100, 1)
    cluster5 = np.repeat(np.expand_dims(logs['New_Prediction'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(9, 12))

    im1 = ax[0].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im2 = ax[1].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_sep,vmin=0,vmax=1)
    im3 = ax[2].imshow(cluster3, interpolation='none', aspect='auto',
                    cmap=cmap_sep,vmin=0,vmax=1)
    im4 = ax[3].imshow(cluster4, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im5 = ax[4].imshow(cluster5, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    ax[5].plot(logs.Entropy,logs.Depth,'-g')


    """
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im2, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    """

    ax[5].set_ylim(ztop,zbot)
    ax[5].invert_yaxis()
    ax[5].grid()
    ax[5].locator_params(axis='x', nbins=3)

    ax[0].set_xlabel("Facies")
    ax[1].set_xlabel("Change layer")
    ax[2].set_xlabel("Predicted change layer")
    ax[3].set_xlabel("Model 1 Prediction")
    ax[4].set_xlabel("Prediction")
    ax[5].set_xlabel('Entropy')

    ax[5].set_xlim(0,1.5)
    
    ax[5].set_yticklabels([])
    ax[0].set_xticklabels([]); ax[0].set_yticklabels([])
    ax[1].set_xticklabels([]); ax[1].set_yticklabels([])
    ax[2].set_xticklabels([]); ax[2].set_yticklabels([])
    ax[3].set_xticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_xticklabels([]); ax[4].set_yticklabels([])

    #f.suptitle('Well: %s'%logs.iloc[0]['Well_Name'], fontsize=14,y=0.94)