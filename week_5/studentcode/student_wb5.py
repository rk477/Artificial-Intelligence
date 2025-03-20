# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here
   
    # get the data from file into a numpy array
    data = np.genfromtxt(datafile_name, delimiter=',')

    # create a K-Means cluster model with  the specified number of clusters
    cluster_model = KMeans(n_clusters=K, n_init=10)
    cluster_model.fit(data)
    cluster_ids = cluster_model.predict(data)

    # create a canvas(fig) and axes to hold your visualisation
    num_features = data.shape[1]
    fig, axs = plt.subplots(num_features, num_features, figsize=(12,12))

    # Get colours for histogram
    hist_col = plt.get_cmap('viridis', K).colors

    # make the visualisation
    # remember to put your user name into the title as specified
    for feat1 in range(num_features):
        axs[feat1, 0].set_ylabel(feature_names[feat1])
        axs[0, feat1].set_xlabel(feature_names[feat1])
        axs[0, feat1].xaxis.set_label_position('top')

        for feat2 in range(num_features):
            x_data = data[:, feat1]
            y_data = data[:, feat2]

            if feat1 != feat2:
                axs[feat1, feat2].scatter(x_data, y_data, c=cluster_ids)
            else:
                # Sort the labels and data so that the classes are in order
                inds = np.argsort(cluster_ids)
                sorted_x_data = x_data[inds]
                sorted_ids = cluster_ids[inds]

                # Split the data into the different classes
                splits = np.split(sorted_x_data, np.unique(sorted_ids, return_index=True)[1][1:])

                # Plot the histogram
                for i, split in enumerate(splits):
                    axs[feat1, feat2].hist(split, bins=20, color=hist_col[i], edgecolor='black', alpha=0.7)


    # title with my name
    fig.suptitle(f'Visualisation of {K} clusters by r6-khadka', fontsize=16, y=0.95)

    # save it to file as specified
    fig.savefig('myVisualisation.jpg')

    # if you don't delete the line below there will be problem!
    # raise NotImplementedError("Complete the function")
    
    return fig,axs
    
    # <==== insert your code above here
    
