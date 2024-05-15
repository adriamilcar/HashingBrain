import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import expon


def plot_ratemaps(ratemaps, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,20), dpi=600)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(ratemaps[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig('plot_ratemaps.jpg', dpi=600)
    plt.show()


def plot_ratemaps_with_centroids(ratemaps, num_fields, centroids, save=False):
    plt.figure(figsize=(20,20), dpi=600)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(ratemaps[i], cmap='hot', origin='lower')
        for j in np.arange(num_fields[i]):
            indx = int(np.sum(num_fields[:i]) + j)
            x, y = centroids[indx]
            plt.scatter(x, y, s=150, c='blue', marker='x')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig('plot_ratemaps_centroids.jpg', dpi=600)
    plt.show()


def plot_place_field_hist(num_fields, save=False):
    '''
    TO DO.
    '''
    place_field_counts = np.histogram(num_fields, bins=np.max(num_fields)+1, density=True)[0]
    plt.figure(figsize=(5,4))
    plt.bar(np.arange(np.max(num_fields)+1), place_field_counts, width=1, color='black', alpha=1, edgecolor='white')
    plt.xlabel('# place fields', fontsize=20)
    plt.ylabel('prob.', fontsize=20)
    plt.yticks(np.linspace(0,1,6), np.linspace(0,1,6).round(1), fontsize=18)
    plt.xticks(np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), fontsize=18)
    plt.ylim(0,1)
    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig('plot_place_field_hist.jpg', dpi=600)
    plt.show()


def plot_polarmaps(p, n_bins=20, n_cells_plot=30, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,16), dpi=600)
    
    for i in range(n_cells_plot):

        bottom = 0.4

        theta = np.linspace(0.0, 2*np.pi, n_bins, endpoint=False)
        radii = p[i]
        width = (2*np.pi) / (n_bins-1)

        ax = plt.subplot(5,6,i+1, polar=True)
        plt.title('cell '+str(i+1))
        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        ax.set_theta_zero_location("W")

        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r / 5.))
            bar.set_alpha(0.8)

    plt.tight_layout()
    if save:
        plt.savefig('plot_polarmaps.jpg', dpi=600)
    plt.show()


def plot_similarity_matrix_clustered(similarity_matrix, cluster_labels, save=False):
    '''
    TO DO.
    '''
    n_samples = similarity_matrix.shape[0]
    sm = similarity_matrix.reshape((n_samples, n_samples, 1)).astype(np.float64)

    plt.imshow(sm[np.argsort(cluster_labels), :][:, np.argsort(cluster_labels)], origin='lower')
    plt.xlim(np.count_nonzero(cluster_labels==-1), cluster_labels.size)
    plt.ylim(np.count_nonzero(cluster_labels==-1), cluster_labels.size)
    plt.colorbar()
    plt.tight_layout()
    if save:
        plt.savefig('plot_similarity_matrix_clustered.jpg', dpi=600)
    plt.show()


def plot_distance_to_wall(dist_to_wall, sizes, norm_area=True, save=False):
    '''
    TO DO.
    '''
    if norm_area:
        C = 20  # n of squares to discretize space
        L = 50  # side length of the arena
        diff_area = (L**2)/C  # the area difference that has to be maintained between subsequent squares

        lengths = np.zeros(C+1)
        lengths[0] = L

        # calculate the side lengths of the subsequent squares
        for i in range(C):
            lengths[i+1] = np.sqrt(np.maximum(lengths[i]**2 - diff_area, 0))

        bin_edges = (L-lengths)/2

        # Alternative: use linear space w.r.t. distance to wall to compute the areas, and normalize later the bin counts by the corresponding areas.
        #bin_edges = np.linspace(0, L/2, C+1) - 0.5
        #bin_areas = np.diff(np.linspace(L, 0, C+1)**2)
    else:
        bin_edges = np.arange(0, np.max(dist_to_wall) + 1, 1) - 0.5

    # Bin the centroids based on their distance to the nearest wall
    bin_counts, _ = np.histogram(dist_to_wall, bins=bin_edges)
    #bin_counts_norm = bin_counts/bin_areas

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(2,1, figsize=(6,6), gridspec_kw={'height_ratios': [1, 4]})

    # Plot the number of centroids in each bin against the average distance to the nearest wall for that bin
    axes[0].bar(bin_centers, bin_counts, width=np.diff(bin_edges), color='black', alpha=1, edgecolor='white')
    #axes[0].set_xlabel('Distance to wall', fontsize=20)
    axes[0].set_ylabel('# place \nfields', fontsize=20)
    axes[0].set_xticks([])
    axes[0].set_yticks([0, np.max(bin_counts).astype(np.int_)], [0, np.max(bin_counts).astype(np.int_)], fontsize=16)
    axes[0].set_xlim(bin_centers[bin_counts!=0][0]-1., np.max(dist_to_wall) + 1.1)
    sb.despine()
    axes[0].set_anchor('W')

    axes[1].scatter(dist_to_wall, sizes, s=4, marker='o', color='black')
    m,b = np.polyfit(dist_to_wall, sizes,1)
    axes[1].plot(dist_to_wall, m*np.array(dist_to_wall)+b, linestyle='-', color='r', linewidth=2)
    #plt.title('corr_coef='+str(np.corrcoef( image_dist, latent_dist )[0][1].round(2)))
    #axes[1].set_title('corr. coef.='+str(spearmanr( dist_to_wall, sizes ).correlation.round(2)))
    axes[1].set_ylabel('place field size (px)', fontsize=20)
    axes[1].set_xlabel('distance to wall (px)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sb.despine()
    axes[1].set_anchor('W')

    plt.tight_layout()
    if save:
        plt.savefig('plot_distance_to_wall.jpg', dpi=600)
    plt.show()


def plot_hyperbolic_geometry(sizes, bin_width=10, save=False):
    '''
    TO DO.
    '''
    # Bin the centroids based on their distance to the nearest wall
    bin_edges = np.arange(np.min(sizes), np.max(sizes) + 1, bin_width)
    bin_counts, _ = np.histogram(sizes, bins=bin_edges)#, density=True)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax1 = plt.subplots()

    left, bottom, width, height = [0.6, 0.41, 0.25, 0.45]
    ax2 = fig.add_axes([left, bottom, width, height])

    # Plot the number of centroids in each bin against the average distance to the nearest wall for that bin
    ax1.bar(bin_centers, bin_counts, width=bin_width, color='black', alpha=.6, edgecolor='white')

    ax2.scatter(bin_centers, bin_counts, s=10, color='black')
    log_bin_counts = np.log(bin_counts, out=np.zeros_like(bin_counts, dtype='float64'), where=(bin_counts!=0))
    log_bin_counts = log_bin_counts[bin_counts!=0]
    bin_centers_ = bin_centers[bin_counts!=0]
    m,b = np.polyfit(bin_centers_, log_bin_counts, 1)
    ax2.plot(bin_centers, np.exp(m*np.array(bin_centers)+b), linestyle='--', color='r', linewidth=2)
    #ax2.set_xticks(np.linspace(60,140,5).astype(np.int_), np.linspace(60,140,5).astype(np.int_), fontsize=9)
    ax2.xaxis.set_tick_params(labelsize=9)
    ax2.set_yscale('log')

    ax1.plot(bin_centers, np.exp(m*np.array(bin_centers)+b), color='black', linewidth=3, linestyle='--')

    ax1.set_xlabel('Field size (px)', fontsize=18)
    ax1.set_ylabel('N fields', fontsize=18)
    ax1.xaxis.set_tick_params(labelsize=16)
    #ax1.set_xticks(np.linspace(60,140,5).astype(np.int_), np.linspace(60,140,5).astype(np.int_), fontsize=16)
    ax1.set_yticks(np.linspace(0, bin_counts.max(), 4).astype(np.int_), np.linspace(0, bin_counts.max(), 4).astype(np.int_), fontsize=16)

    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig('plot_hyperbolic_geometry.jpg', dpi=600)
    plt.show()

