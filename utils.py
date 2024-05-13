import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from scipy.signal import correlate2d
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation
import skdim
from random import shuffle
from PIL import Image
import imagehash
import cmath
from sklearn.decomposition import PCA
import clip
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from itertools import combinations



def load_dataset(directory, task, file_format='.npy', load_position=True, position_filename='position.npy'):
    '''
    Loads all .npy or .jpg images and the pose data per image from a given directory.

    Args:
        directory (str): path to the images to be loaded.
        file_format (str): format of the images. Accepted formats are .npy and .jpg.
        load_position (bool): if True, it will also load the pose data.
        position_filename (str): name of the file with the pose data. The accepted format is .npy.

    Returns:
        images (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) and normalized values [0,1].
        position (2D numpy array): pose data with (x,y) coordinates and angle (in degrees; [0,360]), wit shape (n_samples, 3).
    '''
    ## Load images.
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(file_format) and filename != position_filename:
            filepath = os.path.join(directory, filename)
            if file_format == '.npy':
                image = np.load(filepath)
            elif file_format == '.jpg' or file_format == '.png':
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    images = np.array(images)

    if np.max(images) > 1:   # normalize to [0,1] if values are RGB [0,255].
        images = images/255.

    ## Load position.  
    position = []
    if load_position:
        if position_filename in os.listdir(directory):
            position = np.load(directory+'/'+position_filename)
        else:
            pos_directory = directory+'/pos'
            for i, filename in enumerate(os.listdir(pos_directory)):
                filepath = os.path.join(pos_directory, filename)
                pos = np.load(filepath).tolist()
                position.append(pos)

    pos_cols = (0,2)
    if task == 'openArena':
        pos_cols = (0,1)

    position = np.array(position)[:, pos_cols]

    return images, position


def load_pose_data_from_csv(path):
    '''
    Loads the pose and orientation of the image dataset from a .csv file.

    Args:
        path (str): path to the .csv file containing the position 'X', 'Y', and orientation 'Z'.

    Returns:
        position (2D numpy array): pose data with shape (n_samples, x, y), and normalized values [0,1].
        orientation (2D numpy array): orientation data with shape (n_samples, vec_x, vec_y) and angle values in vectorial form.
        angles (2D numpy array): orientation data with shape (n_samples, theta) and values in radians [0,2pi], where 0 (or 2pi) 
                                 is west and pi/2 is south.
        conjuctive (2D numpy array): pose and orientation data with shape (n_samples, x, y, vec_x, vec_y).
    '''
    data = pd.read_csv(path)

    Xpose = data['X']
    Xpose = Xpose - Xpose.min()
    Ypose = data['Y']
    Ypose = Ypose - Ypose.min()
    position = np.vstack((Xpose, Ypose)).T
    position = position / position.max()

    angles = data['Z'] + np.pi
    n_samples = angles.shape[0]
    orientation = np.zeros((n_samples, 2))
    for i in range(n_samples):
        orientation[i] = [np.cos(angles[i]), np.sin(angles[i])]

    conjunctive = np.hstack((position, orientation))

    return position, orientation, angles, conjuctive


def shuffle_2D_matrix(m):
    '''
    Shuffles a matrix across both axis (not only the first axis like numpy.permutation() or random.shuffle()).

    Args:
        m (2D numpy array): 2D matrix with arbitrary values.

    Returns:
        m_shuffled (2D numpy array): the original matrix 'm', with all the elements shuffled randomly.
    '''
    N = m.size
    ind_shuffled = np.arange(N)
    shuffle(ind_shuffled)
    ind_shuffled = ind_shuffled.reshape((m.shape[0], m.shape[1]))
    ind_x = (ind_shuffled/m.shape[1]).astype(np.int_)
    ind_y = (ind_shuffled%m.shape[1]).astype(np.int_)
    m_shuffled = m[ind_x, ind_y]
    return m_shuffled


def ratemap_filtered_Gaussian(ratemap, std=2):
    '''
    Adds Gaussians filters to a ratemap in order to make it more spatially smooth.

    Args:
        ratemap (2D numpy array): unfiltered ratemap with the activity counts across space.
        std (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units). 

    Returns:
        new_ratemap (2D numpy array): original ratemap filtered with Gaussian smoothing.
    '''
    new_ratemap = gaussian_filter(ratemap, std)   
    return new_ratemap


def ratemaps(embeddings, position, n_bins=50, filter_width=3, occupancy_map=[], n_bins_padding=0):
    '''
    Creates smooth ratemaps from latent embeddings (activity) and spatial position through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        occupancy_map (2D numpy array; default=[]): 2D matrix reflecting the occupancy time across the space, with shape (n_bins+2*n_bins_padding, n_bins+2*n_bins_padding).
        n_bins_padding (int; default=0): the number of extra pixels with 0 value that are added to every side of the arena.

    Returns:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    # Add activation values to each cell in the ratemap and adds Gaussian smoothing.
    n_latent = embeddings.shape[1]
    ratemaps = np.zeros((n_latent, int(n_bins+2*n_bins_padding), int(n_bins+2*n_bins_padding)))
    for i in range(n_latent):
        ratemap_ = np.zeros((n_bins, n_bins))
        for ii, c in enumerate(embeddings[:,i]):
            indx_x = pos_imgs_norm[ii,0]
            indx_y = pos_imgs_norm[ii,1]
            ratemap_[indx_x, indx_y] += c

        if len(occupancy_map) > 0:
            ratemap_ = np.divide(ratemap_, occupancy_map, out=np.zeros_like(ratemap_), where=occupancy_map!=0)
            #ratemap_ = ratemap_/occupancy_map

        ratemaps[i] = np.pad(ratemap_, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)
        if np.any(ratemaps[i]):
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemap_filtered_Gaussian(ratemaps[i], filter_width)
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
        
    return ratemaps


def stats_place_fields(ratemaps, peak_as_centroid=True, min_pix_cluster=0.02, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Runs a simple clustering algorithm to identify place fields, and compute their number, centroids, and sizes, for all ratemaps.

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        peak_as_centroid (bool; default=True): if True, the centroid will be taken as the peak of the place field; if False, it will take the 'center of mass'.
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        all_num_fields (1D numpy array): array with the number of place fields per embedding unit, with shape (n_latent,).
        all_centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        all_sizes (1D numpy array): array with the sizes of all place fields across embedding units, with shape (total_n_place_fields,).
    '''
    all_num_fields = []
    all_centroids = []
    all_sizes = []
    for r in ratemaps:

        ratemap = r.copy()
        
        ## Params.
        total_area = ratemap.shape[0]*ratemaps.shape[1]
        cluster_min = total_area*min_pix_cluster  #50
        cluster_max = total_area*max_pix_cluster  #1250
        
        ## Clustering.
        ratemap[ratemap <  ratemap.max()*active_threshold] = 0
        ratemap[ratemap >= ratemap.max()*active_threshold] = 1

        # First pass of clustering.
        clustered_matrix = np.zeros_like(ratemap)
        current_cluster = 1

        # Go through every bin in the ratemap.
        for yy in range(1,ratemap.shape[0]-1):
            for xx in range(1,ratemap.shape[1]-1):
                if ratemap[  yy, xx ] == 1:
                    # Go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clustered_matrix[ yy+ty, xx+tx ] != 0:
                                clustered_matrix[ yy,xx ] = clustered_matrix[ yy+ty, xx+tx ]

                    if clustered_matrix[ yy, xx ] == 0:
                        clustered_matrix[ yy, xx ] = current_cluster
                        current_cluster += 1
                        
        # Refine clustering: neighbour bins to same cluster number.
        for yy in range(1,clustered_matrix.shape[0]-1):
            for xx in range(1,clustered_matrix.shape[1]-1):
                if clustered_matrix[  yy, xx ] != 0:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clustered_matrix[ yy+ty, xx+tx ] != 0:
                                if clustered_matrix[ yy+ty, xx+tx ] != clustered_matrix[  yy, xx ]:
                                    clustered_matrix[ yy+ty, xx+tx ] = clustered_matrix[  yy, xx ]
                  
        ## Quantify number of place fields.
        clusters_labels = np.delete(np.unique(clustered_matrix), np.where(  np.unique(clustered_matrix) == 0 ) )
        n_place_fields_counter = 0
        clustered_matrix_ = np.copy(clustered_matrix)
        clusters_labels_ = np.copy(clusters_labels)
        for k in range(clusters_labels.size):
            n_bins = np.where(clustered_matrix == clusters_labels[k])[0].size
            if cluster_min <= n_bins <= cluster_max:
                n_place_fields_counter += 1
            else:
                clustered_matrix_[np.where(clustered_matrix_==clusters_labels[k])] = 0
                clusters_labels_ = np.delete(clusters_labels_, np.where(clusters_labels_ == clusters_labels[k]) )

        all_num_fields.append(n_place_fields_counter)
        
        ## Compute centroids.
        centroids = []
        for k in clusters_labels_:
            if peak_as_centroid:  # compute centroid as the peak of the place field.
                x, y = np.unravel_index(np.argmax( r * (clustered_matrix_==k) ), r.shape)
            else:                 # compute the centroid as weighted sum ('center of mass').
                w_x = r[np.where(clustered_matrix_==k)[0], :].sum(axis=1)
                w_x = w_x/w_x.sum()
                x = np.sum(w_x * np.where(clustered_matrix_==k)[0])
                
                w_y = r[:, np.where(clustered_matrix_==k)[1]].sum(axis=0)
                w_y = w_y/w_y.sum()
                y = np.sum(w_y * np.where(clustered_matrix_==k)[1])
            centroids.append([x,y])

        all_centroids += centroids
        
        ## Compute sizes of place fields.
        sizes = []
        for k in clusters_labels_:
            n_bins = np.where(clustered_matrix_ == k)[0].size
            sizes.append(n_bins/total_area)

        all_sizes += sizes
    
    return np.array(all_num_fields), np.array(all_centroids), np.array(all_sizes)


def prop_cells_with_place_fields(ratemaps, min_pix_cluster=0.02, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Computes the number of active (i.e., non-silent) units that have at least one place field.
    
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        prop_cells (float): proportion of active units that have one or more place fields, within the range [0,1].
    '''
    all_num_fields, _, _ = stats_place_fields(ratemaps=clean_ratemaps(ratemaps), min_pix_cluster=min_pix_cluster, 
                                        max_pix_cluster=max_pix_cluster, active_threshold=active_threshold)[0]

    prop_cells = np.count_nonzero(all_num_fields) / all_num_fields.shape[0]

    return prop_cells


def n_active_cells(embeddings):
    '''
    Computes the number of units that have at least some activity (non-silent).
    
    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).

    Returns:
        n_active (int): number of active units.
    '''
    n_active = np.count_nonzero(np.any(embeddings, axis=0))

    return n_active


def polarmaps(embeddings, angles, n_bins=20):
    '''
    Creates polarmaps from embedding activity and angle orientation through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).
        n_bins (int; default=20): resolution of the discretization of angles from which the polarmaps will be computed.

    Returns:
        polarmaps (2D numpy array): 2D matrix containing the polarmaps associated to all embedding units, with 
                                    shape (n_latent, n_bins).
    '''
    # Normalize orientation with respect to resolution to convert orientation to polarmap indices
    orien_imgs_norm = np.copy(angles)
    orien_imgs_norm = orien_imgs_norm/np.max(orien_imgs_norm)
    orien_imgs_norm = orien_imgs_norm*n_bins
    orien_imgs_norm = orien_imgs_norm.astype(int)
    
    # Add activation values to each cell in the ratemap and adds Gaussian smoothing
    n_latent = embeddings.shape[1]
    polarmaps = np.zeros((n_latent, n_bins))
    for i in range(n_latent):
        for ii, c in enumerate(embeddings[:,i]):
            indx = orien_imgs_norm[ii]
            polarmaps[i, indx-1] += c
        if np.any(polarmaps[i]):
            polarmaps[i] = polarmaps[i]/np.max(polarmaps[i])
        
    return polarmaps


def clean_embeddings(embeddings, normalize=False):
    '''
    Takes only the embeddings that are active an any given point, and, if normalize=True, normalizes the values.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        normalize (bool; default=False): if True, the embedding activation values will be normalized to [-1,1], per each unit.

    Returns:
        embeddings_clean (2D numpy array): original embeddings matrix, with the silent units removed, with shape (n_samples, n_active).
    '''
    indxs_active = np.any(embeddings, axis=0)
    n_active = np.sum(indxs_active)
    embeddings_clean = embeddings[:,indxs_active]

    if normalize:  # normalize between [-1, 1].
        n_samples = embeddings.shape[0]
        maxs = np.tile( np.abs(embeddings_clean).max(axis=0), n_samples).reshape((n_samples, n_active))
        embeddings_clean = embeddings_clean / maxs

    return embeddings_clean


def clean_ratemaps(ratemaps):
    '''
    Discards the ratemaps of the silent units.
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).

    Returns:
        ratemaps_clean (3D numpy array): 3D matrix containing the ratemaps associated to the embedding units that are not silent, with shape (n_latent-n_silent, n_bins, n_bins).
    '''
    indxs_active = np.any(ratemaps, axis=(1,2))
    ratemaps_clean = ratemaps[indxs_active]

    return ratemaps_clean


def angular_distance(angle1, angle2):
    '''
    Computes the angular distance between two angles given in radians.

    Args:
        angle1, angle2 (float): angles to compare (in radians).

    Returns:
        dist (float): distance value between angle1 and angle2.
    '''
    delta = np.abs(angle1 - angle2)
    dist = np.min([delta, 2*np.pi - delta])

    return dist


def euclidean_distance(point1, point2):
    '''
    Computes the Euclidean distance (L2 norm) between two points in space.

    Args:
        point1, point2 (1D numpy array): array representing a point in am arbitrary N-dimensional space.

    Returns:
        dist (float): Euclidean distance between point1 and point2 in the N-dimensional space.
    '''
    dist = np.sqrt(np.sum((point1 - point2)**2))

    return dist


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1, vec2 (1D numpy array): Vectors.

    Returns:
        similarity (float): Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


def hashing_data(dataset, in_bits=True):
    '''
    Performs perceptual hashing over all images in the dataset.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        in_bits (bool; default=True): if True, the hashes will be converted to 64 bits; if False, the hashes will be returned as str.

    Returns:
        hashes (1D or 2D numpy array): 1D array with all the hashing values for each image in 'hash' or str format with shape (n_samples,) 
                                       if in_bits=False, or a 2D array with all the hashing values in bits, with shape (n_samples, 64) if in_bits=True.
    '''
    dataset_ = np.copy(dataset)

    if dataset_.shape[1] <= 3:  # if channels are in the second dimension, bring them to the last.
        dataset_ = np.transpose(dataset_, (0,2,3,1))

    if np.max(dataset_) <= 1:
        dataset_ *= 255

    dataset_ = np.uint8(dataset_)

    hashes = np.array([imagehash.phash(Image.fromarray(img)) for img in dataset_])

    n_samples = dataset.shape[0]
    if in_bits:
        hashes = np.array([np.array(hashes[i].hash.flatten()).astype(np.int_).tolist() for i in range(n_samples)])

    return hashes


def hamming_dist_matrix(hashes):
    '''
    Computes Hamming distances between hashes, as well as a corresponding similarity matrix.

    Args:
        hashes (1D or 2D numpy array): 1D array with all the hashing values for each image in 'hash' or str format with shape (n_samples,),
                                       or a 2D array with all the hashing values in bits, with shape (n_samples, 64).

    Returns:
        hash_diffs (2D numpy array): square matrix with the pairwise Hamming distances between all hashes, with shape (n_samples, n_samples).
        similarity_matrix (2D numpy array): square matrix with the pairwise normalized similarity score based on the hash_diffs, with shape (n_samples. n_samples).
    '''
    n_samples = hashes.shape[0]

    hash_diffs = []
    if len(hashes.shape) == 1:  # if hashes are in str values.
        hash_diffs = hashes.reshape((1, n_samples)) - hashes.reshape((n_samples, 1))
    else:  # otherwise they will be in 64-bit values.
        hash_diffs = np.sum(hashes[:, np.newaxis, :] != hashes, axis=2)

    #hash_diffs_norm = hash_diffs / hash_diffs.max()
    #similarity_matrix = 1. - hash_diffs_norm

    similarity_matrix = -hash_diffs

    return hash_diffs, similarity_matrix


def occupancy_map(position, n_bins=50, filter_width=2, n_bins_padding=0):
    '''
    Computes the occupancy map based on the position through time.

    Args:
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        padding_n (int; default=0): the number of extra pixels that are added to every side of the arena.

    Returns:
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    map_occ = np.zeros((n_bins, n_bins))
    for p in pos_imgs_norm:
        ind_x, ind_y = p
        map_occ[ind_x, ind_y] += 1

    map_occ = np.pad(map_occ, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)

    map_occ = ratemap_filtered_Gaussian(map_occ, filter_width)
    map_occ = map_occ/np.sum(map_occ, axis=(0,1))
    occupancy_map = map_occ.T

    return occupancy_map


def spatial_information(ratemaps, occupancy_map):
    '''
    Spatial information score (SI) as computed in Skaggs et al. 1996. The SI is computed per rate (i.e., embedding unit).

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        SI (1D numpy array): array with SI scores, in bit/spike, with shape (n_latent,).
    '''
    ratemaps_ = ratemaps[np.any(ratemaps, axis=(1,2))]
    FR = ratemaps_/(np.mean(ratemaps_, axis=(1,2))[:,np.newaxis,np.newaxis])
    OT = occupancy_map/np.sum(occupancy_map)
    log_FR = np.log2(FR, out=np.zeros_like(FR, dtype='float64'), where=(FR!=0))
    SI = np.sum(FR*OT*log_FR, axis=(1,2))
    
    return SI


def homogeneity_2Dtiling(centroids):
    '''
    Computes a metric of how homogeneously the 2D-space is tiled by the place fields, based on the coefficient of variation of
    the pair-wise minimum distances between centroids (the closer to 0 the better): 
    for all i -> Std( min( d(pos[i], pos[j..k]) ) ) / Mean( min( d(pos[i], pos[j..k]) ) ).

    Args:
        centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).

    Returns:
        cv (float): coefficient of variation of minimum distances across centroids. Values close to 0 indicate homogeneity.
    '''
    n_centroids = centroids.shape[0]
    centroid_dists = []
    for i in range(n_centroids):
        centroid_dists_all = []
        for j in range(i, n_centroids):
            if i != j:
                centroid_dist = euclidean_distance(centroids[i], centroids[j])
                centroid_dists_all.append(centroid_dist)
        if i != j:
            centroid_dists.append( np.min(centroid_dists_all) )
    cv = np.std(centroid_dists) / np.mean(centroid_dists)
    return cv


def dist_to_walls(centroids, occupancy_map):
    '''
    Computes the minimum distance to the arena's walls for every place field centroid.

    Args:
        centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        min_distances (1D numpy array): array with the distance to the nearest wall for every centroid, with shape (total_n_place_fields,).
    '''
    maze = np.copy(occupancy_map)
    maze[ occupancy_map <  occupancy_map.max()*0.1 ] = 0
    maze[ occupancy_map >= occupancy_map.max()*0.1 ] = 1
    maze = maze.T

    # Add padding to the maze and adjust centroids coordinates.
    padded_maze = np.pad(maze, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    adjusted_centroids = centroids + np.array([1, 1])

    # Get the coordinates of all the 0s in the padded maze matrix.
    wall_coords = np.argwhere(padded_maze == 0)
    
    # Compute the Euclidean distances between each adjusted centroid and all the 1s in the padded maze.
    distances = cdist(adjusted_centroids, wall_coords)
    
    # Take the minimum distance for each centroid
    min_distances = np.min(distances, axis=1)
    
    return min_distances


def angles_to_vec(angles):
    '''
    Transforms angles in radians to vectors along the unit-circle.

    Args:
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).

    Returns:
        orientation_vec (2D numpy array): 2D array with the angles converted to vectors in the 2D space, with space (n_samples, 2). 
    '''
    angles_rad = np.copy(angles)
    if np.max(angles) > 2*np.pi:
        angles_rad = np.radians(angles)

    n_samples = angles_rad.shape[0]
    orientation_vec = np.zeros((n_samples, 2))
    for i in range(n_samples):
        orientation_vec[i] = [np.cos(angles_rad[i]), np.sin(angles_rad[i])]

    return orientation_vec


def intrinsic_dimensionality(dataset, method='PCA'):
    '''
    Computes the intrinsic dimensionality of a dataset based on the scikit-dim library. Supported methods are PCA-based and MLE (as in Levina & Bickel (2004)).
    
    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        method (str; default='PCA'): method to estimate the intrinsic dimensionality of the dataset. Valid arguments are 'PCA', and 'MLE', 'TwoNN', and 'FisherS'.

    Returns:
        D (float): intrinsic dimensionality.
    '''
    N = dataset.shape[0]
    X = np.reshape(dataset, (N, -1))

    estimator_D = None
    if method == 'PCA':
        estimator_D = skdim.id.lPCA()
    elif method == 'MLE':
        estimator_D = skdim.id.MLE()
    elif method == 'TwoNN':
        estimator_D = skdim.id.TwoNN()
    elif method == 'FisherS':
        estimator_D = skdim.id.FisherS()
    else:
        print('The specific method is not supported, please check the supported methods in the documentation.')

    D = estimator_D.fit_transform(X)

    return D


def input_output_similarity(dataset, embeddings, N=1e5):
    '''
    Estimates the slope of the correlation between image and embedding similarity, to check whether similar images lead to similar embeddings.
    
    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        N (float; default=1e5): number of random pairs of samples from the dataset and embeddings used to compute the correlation.

    Returns:
        corr_score (float): slope of the spearman's correlation (rank-based) between image similarity and embedding similariy, within the range [0,1].
        slope (float): slope of the linear fit between pairwise dataset distances and embedding distances.
    '''
    spatial_pos_dist = []
    latent_vec_dist = []
    for i in range(int(N)):
        ind_1, ind_2 = np.random.choice(np.arange(dataset.shape[0]), 2, replace=False)
        spatial_pos_dist.append( euclidean_distance(dataset[ind_1], dataset[ind_2]) )
        latent_vec_dist.append( euclidean_distance(embeddings[ind_1], embeddings[ind_2]) )

    slope, _ = np.polyfit(spatial_pos_dist, latent_vec_dist, 1)

    corr_score = spearmanr(spatial_pos_dist, latent_vec_dist).correlation.round(2)

    return corr_score, slope


def population_sparseness(ratemaps, active_threshold=0.2):
    '''
    Estimates the population sparseness as the expected number of active units per pixel (i.e., location in space).
    
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].
    
    Returns:
        sparseness (float): population sparseness score as 1 minus the average proportion of active units across the environment, within the range [0,1].
    '''
    #ratemaps_thres = np.copy(ratemaps)
    ratemaps_thres = clean_ratemaps(ratemaps)
    ratemaps_thres[ratemaps_thres<active_threshold] = 0
    ratemaps_thres[ratemaps_thres>=active_threshold] = 1

    prop_active_per_pixel = np.mean(ratemaps_thres, axis=0)
    sparseness = 1 - np.mean(prop_active_per_pixel)

    return sparseness


def unit_sparseness(embeddings):
    '''
    Computes the expected proportion of samples for which individual units respond to.
    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
    Returns:
        sparseness (float): average proportion of data samples, w.r.t. the total dataset size, that make units be active.
    '''
    embeddings_clean = clean_embeddings(embeddings)
    sparseness = np.mean(np.count_nonzero(embeddings_clean, axis=0)/embeddings_clean.shape[0])

    return sparseness


def allocentricity(embeddings, angles, n_bins=20):
    '''
    Computes an allocentric score as the average of the circular variances (i.e., mean resultant lenghts) of the polarmaps for the non-silent units.
    
    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).
        n_bins (int; default=20): resolution of the discretization of angles from which the polarmaps will be computed.
        
    Returns:
        allocentric_score (float): average circular variance for all non-silent units.
    '''
    all_polarmaps = polarmaps(clean_embeddings(embeddings), angles, n_bins=n_bins)
    bin_centers = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    circ_vars = []
    for polarmap in all_polarmaps:
        mean_vec = np.exp(1j * bin_centers) @ polarmap / np.sum(polarmap)
        circ_var = 1 - np.abs(mean_vec)
        circ_vars.append(circ_var)

    allocentric_score = np.mean(circ_vars)

    return allocentric_score


def linear_decoding_score(embeddings, features, n_baseline=10000):
    '''
    Computes the score of linear regression of embeddings --> features. Features will normally be position (x,y) 
    or orientation (radians or in vectorial form).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        n_baseline (int; default=10000): number of permutation tests (i.e., shuffling the embeddings matrix) to compute the baseline.

    Returns:
        scores (float list): a list with two scores: (1) the evaluation of the linear regression, and (2) an average & std 
                             of n_baseline random permutation tests.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    linear_score = linear_model.score(embeddings, features)

    #baseline
    baselines = []
    for i in range(n_baseline):
        embeddings_shuffled = shuffle_2D_matrix(np.copy(embeddings))
        linear_model_baseline = LinearRegression()
        linear_model_baseline.fit(embeddings_shuffled, features)
        random_score = linear_model_baseline.score(embeddings_shuffled, features)
        baselines.append(random_score)

    baseline_score = [np.mean(baselines), np.std(baselines)]

    ratio = linear_score/(baseline_score[0])

    return linear_score, baseline_score, ratio


def linear_decoding_error(embeddings, features, norm=1):
    '''
    Computes the expected error of a linear decoder that uses the embeddings to predicts features (e.g. position in (x,y)).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        norm (float; default=1): value used to normalize the MSE and bring it to a more convenient scale.

    Returns:
        mean_dist (float): average euclidean distance between the predictions of the decoder and the actual features, normalized by a scalar.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    pred = linear_model.predict(embeddings)

    dist = np.sqrt(np.sum((pred - features)**2, axis=1))
    mean_dist = np.mean(dist) / norm

    return mean_dist


def autocorrelation_2d(ratemaps):
    '''
     Generates the autocorrelation matrices of the 2D ratemaps.
    
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).

    Returns:
        autocorr (3D numpy array): 3D matrix containing the 2D autocorrelation associated to the ratemaps, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    autocorr = []
    for i in range(ratemaps.shape[0]):
        autocorr_map = correlate2d(ratemaps[i], ratemaps[i], mode='same')
        autocorr_map /= ratemaps[i].size
        autocorr.append( autocorr_map )

    return autocorr


def pv_correlation(embeddings1, embeddings2, position, n_bins=50):
    '''
    Computes the population vector (PV) correlation coefficient between two ratemaps (normally corresponding to different epochs).

    Args:
        embeddings1 (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        embeddings2 (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.

    Returns:
        pv_corr (float): average correlation coefficient across spatial locations between the ratemaps corresponding to embeddings 1 and 2.
    '''
    n_total_bins = int(n_bins**2)
    ratemaps1 = ratemaps(embeddings1, position, n_bins=n_bins, filter_width=0, occupancy_map=[], n_bins_padding=0)
    ratemaps1 = np.reshape(ratemaps1, (ratemaps1.shape[0], n_total_bins))

    ratemaps2 = ratemaps(embeddings2, position, n_bins=n_bins, filter_width=0, occupancy_map=[], n_bins_padding=0)
    ratemaps2 = np.reshape(ratemaps2, (ratemaps2.shape[0], n_total_bins))

    corr_coefs = []
    for i in range(n_total_bins):
        corr_coef = pearsonr(ratemaps1[:,i], ratemaps2[:,i])[0]
        corr_coefs.append(corr_coef)

    pv_corr = np.nanmean(corr_coefs).round(3)

    return pv_corr


def image_whitening(dataset):
    '''
    Whitening as in Stringer et al. (2019) Nature.
    '''
    images = dataset
    num_images = images.shape[0]
    image_shape = images[0].shape

    # Compute Fourier spectrum for each image channel
    spectra = [np.abs(np.fft.fft2(img, axes=(0, 1))) for img in images]

    # Average the spectra across images
    avg_spectrum = np.mean(spectra, axis=0)

    whitened_images = []
    for img in images:
        # Compute Fourier transform of the image channels
        img_fft = np.fft.fft2(img, axes=(0, 1))

        # Whitening in the frequency domain
        whitened_fft = img_fft / (avg_spectrum + 1e-8)

        # Transform back to the pixel domain
        whitened_img = np.real(np.fft.ifft2(whitened_fft, axes=(0, 1)))

        # Intensity scaling to match mean and standard deviation
        whitened_mean = np.mean(whitened_img, axis=(0, 1))
        whitened_std = np.std(whitened_img, axis=(0, 1))

        whitened_img = (whitened_img - whitened_mean) * (np.std(img, axis=(0, 1)) / whitened_std) + np.mean(img, axis=(0, 1))
        
        whitened_img = np.clip(whitened_img, 0, 1)
        
        whitened_images.append(whitened_img)

    return np.array(whitened_images)


def zca_image_whitening(image):
    '''
    ZCA whitening of images from first principles.
    '''
    # Reshape the image to a 2D array
    flattened_image = np.reshape(image, (84*84, 3))

    # Compute the covariance matrix
    cov_matrix = np.cov(flattened_image.T)

    # Perform singular value decomposition (SVD) on the covariance matrix
    U, S, V = np.linalg.svd(cov_matrix)

    # Compute the ZCA whitening matrix
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    # Calculate the mean of the flattened image
    mean = np.mean(flattened_image, axis=0)

    # Apply the ZCA whitening transformation
    centered_image = flattened_image - mean
    whitened_image = np.dot(centered_image, zca_matrix.T)

    # Reshape the whitened image back to the original shape
    whitened_image = np.reshape(whitened_image, (84, 84, 3))

    # Normalize the whitened image
    whitened_image -= np.min(whitened_image)
    whitened_image /= np.max(whitened_image)

    return whitened_image


def zca_embeddings_whitening(embeddings):
    '''
    ZCA whitening of embeddings from first principles.
    '''
    features_centered = embeddings - np.mean(embeddings, axis=0)
    covariance_matrix = np.cov(features_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    whitening_matrix = np.dot(np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))), eigenvectors.T)
    whitened_features = np.dot(features_centered, whitening_matrix)
    return whitened_features



def get_powerlaw_exp(embeddings, start_fit=0, cutoff_fit=500, cutoff_dim=None):
    '''
    TO DO.
    '''
    embeddings_reduced = PCA().fit(embeddings)
    variances = embeddings_reduced.explained_variance_ratio_

    if cutoff_dim == None:
        cutoff_dim = variances.shape[0]

    x = np.arange(1, cutoff_dim+1)
    y = variances[:cutoff_dim]

    m, b = np.polyfit(np.log(x[start_fit:cutoff_fit]), np.log(y[start_fit:cutoff_fit]), 1)

    return m, b


def get_indxs_imgs_max_activity(embeddings, max_act_thres=0.8):
    '''
    Returns the indexes of the dataset for which each unit strongly responds to, controlled by an activation and proportionality thresholds.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        max_act_thres (float, default=0.5): proportion of the maximum value from which the unit is considered active or responsive.

    Returns:
        indxs (list): list of data indexes to which every unit responds to, with variable length.
    '''
    # Generate a binary activation matrix based on the threshold
    embeddings_active = np.where(embeddings < max_act_thres * embeddings.max(axis=0), 0, 1)

    # Exclude units where all samples are active
    active_unit_filter = np.mean(embeddings_active, axis=0) < 1
    embeddings_active_specific = embeddings_active[:, active_unit_filter].T

    # Collect indices of active samples for each feature
    indxs = [ np.nonzero(unit_activations)[0] for unit_activations in embeddings_active_specific ]

    return indxs


def encode_images(dataset, network='CLIP'):
    '''
    Encodes a dataset of images using a specified neural network. The function supports
    encoding through three different networks: CLIP, VGG-16, and Inception V3. It handles
    image preprocessing, model loading, and feature extraction, returning a numpy array
    of encoded features.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) or
                                  shape (n_samples, n_pixels_height, n_pixels_width, n_channels). Values can be in the ranges
                                  [0,1] or [0,255].
        network (str, default='CLIP'): The name of the network to use for encoding. Supported values are 'CLIP', 'VGG', and 'Inception'.

    Returns:
        numpy.ndarray: An array of encoded features. Each row corresponds to the features
                       extracted from an image.

    Raises:
        ValueError: If the specified network is not supported.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup model and transformations based on the selected network
    if network == 'CLIP':
        model, preprocess = clip.load('ViT-B/32', device=device)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    elif network == 'VGG':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Identity()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif network == 'Inception':
        model = models.inception_v3(pretrained=True, aux_logits=True)
        model.fc = torch.nn.Identity()
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unsupported network type specified.")

    model.eval()
    model.to(device)

    class ImageDataset(Dataset):
        def __init__(self, images, transform):
            self.images = images
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            image = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype != np.uint8 else Image.fromarray(image)
            image = self.transform(image)
            return image

    # Create dataset and dataloader
    batch_size = 64
    dataset = ImageDataset(dataset, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract features
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            if network == 'CLIP':
                outputs = model.encode_image(images)
            else:
                outputs = model(images)
            features.extend(outputs.cpu().numpy())

    return np.array(features)


def reduce_dimensionality(features, method='UMAP'):
    '''
    Reduces the dimensionality of the given feature matrix to 2D using specified method.

    Parameters:
    - features (2D numpy array): The high-dimensional feature matrix to reduce.
    - method (str): The dimensionality reduction method to use ('UMAP', 'TSNE', 'MDS').
                     Default is 'UMAP'.

    Returns:
    - reduced (2D numpy array): A 2D array where each row represents the 2D projection of the corresponding
                     high-dimensional feature.

    Raises:
    - ValueError: If an unsupported method is specified.
    '''
    reduced = []
    if method.upper() == 'TSNE':
        # t-SNE
        tsne = TSNE(n_components=2)
        reduced = tsne.fit_transform(features)
    elif method.upper() == 'MDS':
        # MDS
        mds = MDS(n_components=2)
        reduced = mds.fit_transform(features)
    elif method.upper() == 'UMAP':
        # UMAP
        umap_fit = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1)
        reduced = umap_fit.fit_transform(features)
    else:
        raise ValueError("Unsupported method: {}. Use 'UMAP', 'TSNE', or 'MDS'.".format(method))

    return reduced


def build_hulls(embeddings, images_2d, max_act_thres=0.8):
    '''
    Analyze feature activations to determine clusters and calculate overlap metrics.
    
    Parameters:
        embeddings (np.array): The embedding matrix with shape (n_samples, n_features).
        images_2d (np.array): The 2D coordinates of samples with shape (n_samples, 2).
        max_act_thres (float, default=0.8): Proportionality threshold for activation.
    
    Returns:
        float: The average overlap metric for all convex hulls.
        list: A list of polygons representing the convex hulls for plotting.
    '''

    # Determine the number of active units
    n_active_units = n_active_cells(embeddings)

    # Get indexes of images that most strongly activate each unit.
    indxs = get_indxs_imgs_max_activity(embeddings, max_act_thres=max_act_thres)

    # Clustering and convex hull creation
    hull_polygons = []
    for img_indxs in indxs:
        points = images_2d[img_indxs]
        clustering = DBSCAN(eps=1, min_samples=4).fit(points)
        labels = clustering.labels_
        
        for k in set(labels):
            if k != -1:
                cluster_points = points[labels == k]
                if len(cluster_points) >= 3:
                    hull = ConvexHull(cluster_points)
                    polygon = Polygon(cluster_points[hull.vertices])
                    hull_polygons.append(polygon)

    # Calculate the average Intersection over Union (IoU) for all pairs of convex hulls
    total_intersection_over_union = 0
    pair_count = 0
    for i, j in combinations(range(len(hull_polygons)), 2):
        intersection = hull_polygons[i].intersection(hull_polygons[j]).area
        union = hull_polygons[i].union(hull_polygons[j]).area
        total_intersection_over_union += intersection / union if union > 0 else 0
        pair_count += 1

    average_overlap_metric = total_intersection_over_union / pair_count if pair_count > 0 else 0
    
    return average_overlap_metric, hull_polygons

