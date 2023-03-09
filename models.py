import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Conv_AE(nn.Module):
    '''
    TO DO
    '''
    def __init__(self, n_hidden=100, img_env='animalai'):
        super().__init__()

        if img_env == 'weebots':
            dim1, dim2 = 30, 40
        elif img_env == 'animalai':
            dim1, dim2 = 11, 11

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * dim1 * dim2, n_hidden, dtype=torch.float32)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 128 * dim1 * dim2, dtype=torch.float32)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * dim1 * dim2)
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x))
        x = x.view(-1, 128, dim1, dim2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(model, train_loader, num_epochs=1000, learning_rate=1e-3, device='cpu'):
    '''
    TO DO
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                pbar.update(1)
                pbar.set_description( f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}" )

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


def predict(image, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().
    Args:
        image (3D numpy array): sample image with shape (n_channels, n_pixels_height, n_pixels_width).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.
    Returns:
        output_img (3D numpy array): output image with shape (n_pixels_height, n_pixels_width, n_channels)
    '''
    if image.shape[-1] == 3:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image).detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img

def get_latent_vectors(dataset, model):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.
    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.
    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] == 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            latent = model.encoder(inputs.to('cuda'))
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors

    