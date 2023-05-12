import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def KL_div(p, q):
    log_ = torch.where(q != 0, torch.log(p/q), torch.zeros_like(q))
    return p*log_ + (1-p)*torch.log((1-p)/(1-q))


class Conv_AE(nn.Module):
    def __init__(self, n_hidden=100, hard_sparsity=False, soft_sparsity=False, k=1, hard_sparsity_min_epochs=0, hidden_constraints=[]):
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (84,84,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
            sparsity (bool; default=False): if True, sparsity of proportion k is added to the middle layer during forward pass.
            k (float; default=1): if sparsity=True, k is the proportion of active neurons allowed at once, within the range [0,1].
        '''
        super().__init__()

        self.hard_sparsity = hard_sparsity
        self.kth_percentile = int((1-k) * n_hidden) + 1
        self.hard_sparsity_min_epochs = hard_sparsity_min_epochs
        self.current_epoch = 0
        if self.hard_sparsity_min_epochs > 0:
            ks = np.linspace(1, k, hard_sparsity_min_epochs)
            self.kth_percentiles = ((1-ks) * n_hidden).astype(int) + 1

        self.soft_sparsity = soft_sparsity
        self.sparsity_proportion = k            # desired: 0.12

        self.hidden_constraints = hidden_constraints

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 10, 10

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)  
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        if self.hard_sparsity:
            if (self.current_epoch < self.hard_sparsity_min_epochs) and (self.hard_sparsity_min_epochs > 0):
                kth = self.kth_percentiles[self.current_epoch]
            else:
                kth = self.kth_percentile
            thres = torch.kthvalue(h, kth, keepdim=True)[0]
            h = torch.where(h >= thres, h, torch.zeros_like(h))
            h.requires_grad_(True).retain_grad()
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, L1_lambda=0, alpha=0, soft_sparsity_weight=0, epoch=0):
        self.current_epoch = epoch

        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        '''
        # To regularize the weights instead of the activations.
        weights = self.fc1.weight
        Gram = torch.mm(weights, weights.t())
        I = torch.eye(weights.shape[0], device='cuda')
        '''

        batch_size, hidden_dim = hidden.shape
        
        if len(self.hidden_constraints) > 0:
            a, b = self.hidden_constraints
            Gram = torch.mm(hidden.t(), hidden)        # Compute the Gramian matrix of the hidden layer's activations (pairwise-correlations or similarity).
            I = torch.eye(hidden_dim, device='cuda')   # Identity matrix.
            M = (a - Gram) * I + (b - Gram) * (1 - I)

        else:
            M = torch.Tensor(0)

        hidden_constraint_loss = alpha * torch.norm(M) / (batch_size*hidden_dim)

        l1_penalty = L1_lambda * hidden.abs().sum() / (batch_size*hidden_dim)

        if self.hard_sparsity:
            sparse_mask = torch.where(hidden != 0, torch.ones_like(hidden), torch.zeros_like(hidden))
            hidden.register_hook(lambda grad: grad * sparse_mask)

        sparsity_loss = 0.
        if self.soft_sparsity:  # sparsity penalty so that the loss increases if the fraction of units active at each datapoint deviates from a desired one (0.12).
            '''
            hidden_active = torch.where(hidden > 1e-3, hidden, torch.zeros_like(hidden))
            hidden_active_prop = torch.count_nonzero(hidden_active, axis=1) / hidden.shape[1]
            diff = hidden_active_prop - self.sparsity_proportion
            sparsity_loss = soft_sparsity_weight * torch.norm(diff)
            '''
            mean_activation = torch.mean(hidden, dim=0)
            sparsity_loss = soft_sparsity_weight * torch.sum(KL_div(self.sparsity_proportion, mean_activation))

        loss = recon_loss + hidden_constraint_loss + l1_penalty + sparsity_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()
    

class Conv_VAE(nn.Module):
    def __init__(self, n_hidden=100):
        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 10, 10

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden*2)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)  
        x = F.relu(self.fc1(x))
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def backward(self, optimizer, criterion, x, y_true, L1_lambda=0, orth_alpha=0, soft_sparsity_weight=0, epoch=0):
        optimizer.zero_grad()
        y_pred, mu, logvar = self.forward(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()


def create_dataloader(dataset, batch_size=128, reshuffle_after_epoch=True):
    '''
    Creates a DataLoader for Pytorch to train the autoencoder with the image data converted to a tensor.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        batch_size (int; default=32): the size of the batch updates for the autoencoder training.

    Returns:
        DataLoader (Pytorch DataLoader): dataloader that is ready to be used for training an autoencoder.
    '''
    if dataset.shape[-1] == 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


def train_autoencoder(model, train_loader, dataset=[], num_epochs=100, learning_rate=1e-3, L2_weight_decay=0, L1_lambda=0, alpha=0, soft_sparsity_weight=0):
    '''
    TO DO.
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_weight_decay)
    criterion = nn.MSELoss()

    model = model.to('cuda')

    history = []
    embeddings = []
    if len(dataset) > 0:
        embeddings = [ get_latent_vectors(dataset=dataset, model=model) ]
    for epoch in range(num_epochs):
        running_loss = 0.
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to('cuda')

                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, L1_lambda=L1_lambda, 
                                      alpha=alpha, soft_sparsity_weight=soft_sparsity_weight, epoch=epoch)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        history.append(running_loss/len(train_loader))

        if len(dataset) > 0:
            embeddings.append( get_latent_vectors(dataset=dataset, model=model) )

    embeddings = np.array(embeddings)

    return history, embeddings


def predict(image, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().

    Args:
        image (3D numpy array): sample image with shape (n_channels, n_pixels_height, n_pixels_width).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.

    Returns:
        output_img (3D numpy array): output image with shape (n_pixels_height, n_pixels_width, n_channels)
    '''
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image)[0].detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img


def get_latent_vectors(dataset, model, batch_size=128):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0,3,1,2))
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


def find_max_activation_images(model, img_shape=[3, 84, 84]):
    '''
    To be tested.
    '''
    images = []
    for i in range(model.n_hidden):
        # Initialize input image
        x = torch.randn(1, img_shape[0], img_shape[1], img_shape[2], device='cuda', requires_grad=True)

        # Use optimizer to perform gradient ascent
        optimizer = optim.Adam([x], lr=1e-3)

        for j in range(1000):
            optimizer.zero_grad()
            _, mu, _ = model(x)
            loss = -mu[0, i]  # maximize activation of ith unit
            loss.backward()
            optimizer.step()

        # Add image to list
        images.append(x.detach().cpu().numpy()[0, 0])

    return np.array(images)
    