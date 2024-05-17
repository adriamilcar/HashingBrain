import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def generate_varying_freq_signal(min_freq, max_freq, sampling_freq, total_time):
    # Create an array of time points
    t = np.linspace(0, total_time, int(sampling_freq * total_time))

    # Linearly interpolate the frequency from min_freq to max_freq
    instantaneous_freq = np.linspace(min_freq, max_freq, len(t))

    # Integrate the instantaneous frequency to get the phase
    phase = np.cumsum(instantaneous_freq) / sampling_freq

    # Generate the signal
    signal = np.sin(2 * np.pi * phase)

    return t, signal, instantaneous_freq


def sliding_window_dataset(signal, t_sample, sampling_freq):
    # Calculate the number of points in each sample
    sample_points = int(t_sample * sampling_freq)

    # Calculate the total number of samples we can extract
    num_samples = len(signal) - sample_points + 1

    # Initialize the dataset
    data = np.zeros((num_samples, sample_points))

    # Apply the sliding window
    for i in range(num_samples):
        data[i] = signal[i:i + sample_points]
        
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return data


def deterministic_gaussian_sampling(signal, t, instantaneous_freq, mean_freq, std_freq, sampling_freq, window_size_seconds):
    # Calculate the number of points in each sample
    sample_points = int(window_size_seconds * sampling_freq)
    # Calculate the total number of samples to match what we would get from the sliding window approach (uniform distribution)
    num_samples = len(signal) - sample_points + 1
    
    # Define the Gaussian distribution over the frequency range
    x = np.linspace(min_freq, max_freq, len(instantaneous_freq))
    gaussian_pdf = stats.norm.pdf(x, mean_freq, std_freq)
    gaussian_cdf = np.cumsum(gaussian_pdf)
    gaussian_cdf /= gaussian_cdf[-1]  # Normalize to form a proper CDF

    # Sample indices using the inverse CDF
    random_values = np.linspace(0, 1, num_samples + 2)[1:-1]  # exclude exact 0 and 1 for stability
    sampled_indices = np.searchsorted(gaussian_cdf, random_values).astype(int)
    
    # Convert window size in seconds to number of samples
    window_size_samples = int(window_size_seconds * sampling_freq)

    # Dataset to store the samples
    data = np.zeros((num_samples, window_size_samples))

    for i, index in enumerate(sampled_indices):
        start_index = max(0, index - window_size_samples // 2)
        end_index = start_index + window_size_samples
        if end_index > len(signal):
            start_index = len(signal) - window_size_samples
            end_index = len(signal)
        data[i] = signal[start_index:end_index]

    # Reshape the dataset to add an extra dimension
    data = data.reshape(num_samples, window_size_samples, 1)

    return data


def create_dataloader(data, batch_size=256, reshuffle_after_epoch=True):
    '''
    Creates a DataLoader for Pytorch to train the autoencoder with the image data converted to a tensor.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        batch_size (int; default=32): the size of the batch updates for the autoencoder training.

    Returns:
        DataLoader (Pytorch DataLoader): dataloader that is ready to be used for training an autoencoder.
    '''
    if data.shape[-1] == 1:
        data = np.transpose(data, (0,2,1))
    tensor_dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(data).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


class ConvAE(nn.Module):
    def __init__(self, n_hidden=100, sample_length=1000, kernel_size=100):
        super().__init__()

        self.n_hidden = n_hidden
        self.kernel_size=kernel_size

        # Encoder
        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=1)
        
        # Hidden layer
        self.flattened_size = self.calculate_flattened_size(sample_length, kernel_size)
        self.fc1 = nn.Linear(64 * self.flattened_size, n_hidden)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.flattened_size)
        self.conv4 = nn.ConvTranspose1d(64, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=0)
        self.conv5 = nn.ConvTranspose1d(32, 16, kernel_size=kernel_size, stride=2, padding=1, output_padding=0)
        self.conv6 = nn.ConvTranspose1d(16, 1, kernel_size=kernel_size, stride=2, padding=1, output_padding=0)

    def calculate_flattened_size(self, sample_length, kernel_size):
        size_after_conv1 = (sample_length - kernel_size + 2 * 1) // 2 + 1
        size_after_conv2 = (size_after_conv1 - kernel_size + 2 * 1) // 2 + 1
        size_after_conv3 = (size_after_conv2 - kernel_size + 2 * 1) // 2 + 1
        return size_after_conv3

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self.flattened_size)
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, self.flattened_size)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, alpha=0):
        optimizer.zero_grad()
        y_pred, hidden = self.forward(x)
        recon_loss = criterion(y_pred, y_true)

        whitening_loss = 0
        batch_size, hidden_dim = hidden.shape
        if alpha != 0:
            C = torch.eye(hidden_dim, device='cuda') - torch.mm(hidden.t(), hidden)
            whitening_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
        loss = recon_loss + whitening_loss
        loss.backward()
        optimizer.step()
        
        return recon_loss.item()


def train_autoencoder(model, train_loader, num_epochs=1000, learning_rate=1e-3, alpha=0):
    """
    Train the autoencoder model.
    :param model: Autoencoder model.
    :param dataloader: DataLoader for the training data.
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate for the optimizer.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to('cuda')
    history = []
    for epoch in range(num_epochs):
        running_loss = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to('cuda')

                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        history.append(running_loss/len(train_loader))
        
    return history


def predict(sample, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().

    Args:
        sample (3D numpy array): sample with shape (n_channels, n_timepoints).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.

    Returns:
        output_sample (3D numpy array): output timeseries with shape (n_timepoints, n_channels)
    '''
    if sample.shape[-1] <= 4:
        sample = np.transpose(sample, (1,0))
    n_channels, n_timepoints = sample.shape
    sample = np.reshape(sample, (1, n_channels, n_timepoints))
    sample = torch.from_numpy(sample).float().to(next(model.parameters()).device)
    output_sample = model(sample)[0].detach().cpu().numpy()
    output_sample = np.reshape(output_sample, (n_channels, n_timepoints))
    output_sample = np.transpose(output_sample, (1,0))
    return output_sample


def create_test_dataset(freq_range, delta_freq, sample_length):
    """
    Build a test dataset with samples at different frequencies.
    Args:
        min_freq (float): Minimum frequency.
        max_freq (float): Maximum frequency.
        delta_freq (float): Frequency resolution.
        sample_length (int): Length of each sample.
    Returns:
        torch.Tensor: Test dataset.
    """
    frequencies = np.arange(freq_range[0], freq_range[1], delta_freq)
    test_dataset = []
    for freq in frequencies:
        t = np.linspace(0, 1, sample_length, endpoint=False)
        sample = np.sin(2 * np.pi * freq * t)
        test_dataset.append(sample)
    test_dataset = np.reshape(test_dataset, (len(test_dataset), len(test_dataset[0]), 1))
    return test_dataset


def clean_embeddings(embeddings):
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
    return embeddings_clean


def get_embeddings(dataset, model, batch_size=256):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] == 1:
        dataset = np.transpose(dataset, (0,2,1))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            latent = model(inputs.to('cuda'))[1]
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors


def normalize_and_reorder(arr):
    """
    This function takes a 2D numpy array and performs two operations:
    1. Normalizes each row of the array based on its maximum value.
    2. Reorders the rows based on the index of the peak (maximum) value in each row.

    Parameters:
    arr (numpy.ndarray): The input 2D numpy array.

    Returns:
    numpy.ndarray: The processed 2D numpy array.
    """
    # Normalize each row by its maximum value
    row_maxes = np.max(arr, axis=1).reshape(-1, 1)
    normalized_arr = arr / row_maxes

    # Find the index of the maximum value in each row
    peak_indices = np.argmax(normalized_arr, axis=1)

    # Reorder the rows based on the peak indices
    sorted_indices = np.argsort(peak_indices)
    reordered_arr = normalized_arr[sorted_indices]

    return reordered_arr


def extract_feature_images(model, embeddings, clamping_value='max_unit'):
    '''
    TO DO.
    '''
    indxs_active = np.arange(embeddings.shape[1])[np.any(embeddings, axis=0)]
    images = []
    for i in np.arange(model.n_hidden):
        if i in indxs_active:
            input_ = torch.zeros(model.n_hidden).to('cuda')
            activations = torch.tensor(embeddings[:,i])
            if clamping_value == 'max_unit':
            	clamp_value = torch.max(activations).item()
            elif clamping_value == 'mean':
            	clamp_value = torch.mean(torch.max(torch.tensor(embeddings), dim=0)[0])
            input_[i] = clamp_value
            img = model.decoder(input_)[0].detach().cpu().numpy().T
            images.append(img.tolist())
        #else:
            #img = np.zeros(embeddings.shape[0])
            #images.append(img.tolist())
    images = np.array(images)
    
    return images
