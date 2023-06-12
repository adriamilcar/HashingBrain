
class Conv_AE(nn.Module):
    def __init__(self, n_hidden=100, hidden_regularization=False): #, hard_sparsity=False, soft_sparsity=False, k=1, hard_sparsity_min_epochs=0):
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (84,84,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
            sparsity (bool; default=False): if True, sparsity of proportion k is added to the middle layer during forward pass.
            k (float; default=1): if sparsity=True, k is the proportion of active neurons allowed at once, within the range [0,1].
        '''
        super().__init__()

        '''
        self.hard_sparsity = hard_sparsity
        self.kth_percentile = int((1-k) * n_hidden) + 1
        self.hard_sparsity_min_epochs = hard_sparsity_min_epochs
        self.current_epoch = 0
        if self.hard_sparsity_min_epochs > 0:
            ks = np.linspace(1, k, hard_sparsity_min_epochs)
            self.kth_percentiles = ((1-ks) * n_hidden).astype(int) + 1

        self.soft_sparsity = soft_sparsity
        self.sparsity_proportion = k            # desired: 0.12
        '''
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
        '''
        if self.hard_sparsity:
            if (self.current_epoch < self.hard_sparsity_min_epochs) and (self.hard_sparsity_min_epochs > 0):
                kth = self.kth_percentiles[self.current_epoch]
            else:
                kth = self.kth_percentile
            thres = torch.kthvalue(h, kth, keepdim=True)[0]
            h = torch.where(h >= thres, h, torch.zeros_like(h))
            h.requires_grad_(True).retain_grad()
        '''
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, alpha=0, beta=0): #, L1_lambda=0, soft_sparsity_weight=0, epoch=0):
        #self.current_epoch = epoch

        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        '''
        # To regularize the weights instead of the activations.
        weights = self.fc1.weight
        Gram = torch.mm(weights, weights.t())
        I = torch.eye(weights.shape[0], device='cuda')
        '''
        
        # Whitening loss (soft batch whitening).
        hidden_reg_loss = 0
        sparsity_loss = 0
        batch_size, hidden_dim = hidden.shape
        if self.hidden_regularization:

            # SSCP matrix
            #M = torch.mm(hidden.t(), hidden)

            # Covariance matrix
            hidden_centered = hidden - torch.mean(hidden, dim=0, keepdim=True)
            M = torch.mm(hidden_centered.t(), hidden_centered) / (batch_size-1)

            I = torch.eye(hidden_dim, device='cuda')
            lambda_ = 1 #0.1
            C = lambda_*I - M   # whitening --> generates spatial tuning
            #C = M * (1 - I)    # decorrelation --> does not generate spatial tuning
            #C = I - M * I      # standarization --> does not generate spatial tuning
            hidden_reg_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
            sparsity_loss = beta * hidden.sum() / (batch_size*hidden_dim)  # L1 regularization

        '''
        if self.hard_sparsity:
            sparse_mask = torch.where(hidden != 0, torch.ones_like(hidden), torch.zeros_like(hidden))
            hidden.register_hook(lambda grad: grad * sparse_mask)

        sparsity_loss = 0.
        if self.soft_sparsity:  # sparsity penalty so that the loss increases if the fraction of units active at each datapoint deviates from a desired one (0.12).
            #hidden_active = torch.where(hidden > 1e-3, hidden, torch.zeros_like(hidden))
            #hidden_active_prop = torch.count_nonzero(hidden_active, axis=1) / hidden.shape[1]
            #diff = hidden_active_prop - self.sparsity_proportion
            #sparsity_loss = soft_sparsity_weight * torch.norm(diff)
            mean_activation = torch.mean(hidden, dim=0)
            sparsity_loss = soft_sparsity_weight * torch.sum(KL_div(self.sparsity_proportion, mean_activation))
        '''

        loss = recon_loss + hidden_reg_loss + sparsity_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()