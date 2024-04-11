import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils import *


def RAI(fan_in, fan_out):
    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1]
    b = V[:, -1]
    return W, b


class Conv_AE(nn.Module):
    def __init__(self, n_hidden=100, hidden_regularization=True):
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (84,84,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
        '''
        super().__init__()

        self.hidden_regularization = hidden_regularization

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

        self.apply_custom_initialization()

    def apply_custom_initialization(self):
        fan_in, fan_out = self.fc1.weight.data.size(1), self.fc1.weight.data.size(0)
        W, b = RAI(fan_in, fan_out)
        self.fc1.weight.data = torch.from_numpy(W.T).float()
        self.fc1.bias.data = torch.from_numpy(b).float()

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
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, alpha=0, beta=0, gamma=0):

        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        # Whitening loss (soft batch whitening).
        whitening_loss = 0
        sparsity_loss = 0
        variability_loss = 0
        batch_size, hidden_dim = hidden.shape
        if self.hidden_regularization:

            if alpha != 0:
                # SSCP matrix
                M = torch.mm(hidden.t(), hidden)

                # Covariance matrix
                #hidden_centered = hidden - torch.mean(hidden, dim=0, keepdim=True)
                #M = torch.mm(hidden_centered.t(), hidden_centered) / (batch_size-1)

                I = torch.eye(hidden_dim, device='cuda')
                lambda_ = .1 #0.1
                C = lambda_*I - M   # orthonormality (whitening?) --> generates spatial tuning
                #C = M * (1 - I)    # decorrelation --> does not generate spatial tuning
                #C = I - M * I      # standarization --> does not generate spatial tuning
                whitening_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)   # change to /(hidden_dim**2)
        
            if beta != 0:
                #sparsity_loss = beta * torch.norm(hidden, 1) / (batch_size*hidden_dim)  # L1 regularization
                sparsity_loss = beta * torch.sum(torch.abs(hidden)) / (batch_size*hidden_dim)

            if gamma != 0:
                variability_loss = -gamma * torch.sum(torch.var(hidden, dim=0)) / hidden_dim

        loss = recon_loss + whitening_loss + sparsity_loss + variability_loss
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

    def backward(self, optimizer, criterion, x, y_true, alpha=0): #, L1_lambda=0, soft_sparsity_weight=0, epoch=0):
        optimizer.zero_grad()
        y_pred, mu, logvar = self.forward(x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = criterion(y_pred, y_true) + kl_div
        loss.backward()
        optimizer.step()
        return loss.item()


def create_dataloader(dataset, batch_size=256, reshuffle_after_epoch=True):
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


def train_autoencoder(model, train_loader, dataset=[], model_latent=None, num_epochs=1000, learning_rate=1e-4, alpha=2e3, beta=0, gamma=0, L2_weight_decay=0):
    '''
    TO DO.
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_weight_decay)
    criterion = nn.MSELoss()

    model = model.to('cuda')

    history = []
    #embeddings = []
    powerlaw_scores = []
    intrinsic_dims = []
    event_memory_scores = []
    if len(dataset) > 0:
        #embeddings = [ get_latent_vectors(dataset=dataset, model=model) ]
        embeddings = get_latent_vectors(dataset=dataset, model=model)
        powerlaw_scores = [ get_powerlaw_exp(embeddings) ]
        intrinsic_dims = [ intrinsic_dimensionality(embeddings, method='PCA') ]
        if model_latent != None:
            event_memory_scores = [ event_memories_quality(model, model_latent, dataset, clamping_value=0.4) ]
    for epoch in range(num_epochs):
        running_loss = 0.
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to('cuda')

                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, beta=beta, gamma=gamma)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        history.append(running_loss/len(train_loader))

        if len(dataset) > 0:
            #embeddings.append( get_latent_vectors(dataset=dataset, model=model) )
            embeddings = get_latent_vectors(dataset=dataset, model=model)
            powerlaw_scores.append( get_powerlaw_exp(embeddings) )
            intrinsic_dims.append( intrinsic_dimensionality(embeddings, method='PCA') )
            if model_latent != None:
                event_memory_scores.append( event_memories_quality(model, model_latent, dataset, clamping_value=0.4) )

    #embeddings = np.array(embeddings)

    return history, powerlaw_scores, intrinsic_dims, event_memory_scores


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


def get_predictions(train_loader, model, loss_criterion=nn.MSELoss()):

    model.eval()

    criterion = loss_criterion

    all_preds = []
    with torch.no_grad():
        total_loss = 0.
        for inputs, _ in train_loader:
            inputs = inputs.to('cuda')

            predictions = model(inputs)[0]
            all_preds.append(predictions.cpu().numpy())

            loss = criterion(predictions, inputs)
            total_loss += loss.item() * inputs.size(0)

    all_preds = np.concatenate(all_preds)
    average_loss = total_loss / len(train_loader.dataset)

    return all_preds, average_loss


def get_latent_vectors(dataset, model, batch_size=256):
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
    #model = model.to('cuda')
    for i in range(model.n_hidden):
        # Initialize input image
        x = torch.randn(1, img_shape[0], img_shape[1], img_shape[2], device='cuda', requires_grad=True)

        # Use optimizer to perform gradient ascent
        optimizer = optim.Adam([x], lr=1e-3)

        for j in range(1000):
            optimizer.zero_grad()
            _, mu = model(x)
            loss = -mu[0, i]  # maximize activation of ith unit
            loss.backward()
            optimizer.step()

        # Add image to list
        images.append(x.detach().cpu().numpy()[0, 0])

    return np.array(images)
    

def extract_feature_images(model, embeddings, clamping_value=None, input_dims=[84,84,3]):
    '''
    TO DO. Choice of the appropriate clamping value to be fixed --> mean doesn't seem to work well.
    '''
    indxs_active = np.arange(embeddings.shape[1])[np.any(embeddings, axis=0)]
    images = []
    for i in np.arange(model.n_hidden):
        if i in indxs_active:
            input_ = torch.zeros(model.n_hidden).to('cuda')
            activations = torch.tensor(embeddings[:,i])
            clamp_value = torch.mean(activations[torch.nonzero(activations)])
            if clamping_value != None:
                clamp_value = clamping_value
            input_[i] = clamp_value
            img = np.transpose( model.decoder(input_)[0].detach().cpu().numpy(), (1,2,0) )
            images.append(img)
        else:
            img = np.zeros(input_dims)
            images.append(img)
    images = np.array(images)
    
    return images


def event_memories_quality(model, model_latent, dataset, clamping_value=None, input_dims=[84,84,3]):
    '''
    Gets an estimate of how similar the images generated by the middle units in ""model" are to the images in the original dataset.
    '''
    # Generated the images being encoded by all single units.
    embeddings = get_latent_vectors(dataset, model)
    output_images = extract_feature_images(model, embeddings, clamping_value, input_dims)
    indxs_nonempty = np.any(output_images, axis=(1,2,3))
    output_images = output_images[indxs_nonempty]

    # Get latent space vector of each decoded image based on unregularised autoencoder (preserves input-output similarity).
    output_images_latent = get_latent_vectors(output_images, model_latent)

    # Compute pairwise distances (Euclidean)
    embeddings_dataset = get_latent_vectors(dataset, model_latent)
    distances = cdist(output_images_latent, embeddings_dataset, 'euclidean')
    
    # Compute similarity metric (e.g., mean minimum distance)
    mean_min_distance = distances.min(axis=1).mean()
    
    return mean_min_distance



def RAI(fan_in, fan_out):
    '''
    Randomized asymmetric initializer. It draws samples using RAI where fan_in is the number of input units in the weight
    tensor and fan_out is the number of output units in the weight tensor.
    '''
    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1].T
    b = V[:, -1]
    return W.astype(np.float32), b.astype(np.float32)

