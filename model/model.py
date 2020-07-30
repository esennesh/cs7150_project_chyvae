import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ShapesChyVae(BaseModel):
    def __init__(self, z_dim=10, data_dim=28*28):
        super().__init__()
        self._z_dim = z_dim
        self._data_dim = data_dim

        self.encoder_convs = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 1), nn.ReLU(),
        )
        self.encoder_linears = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU(),
        )
        self.mu_encoder = nn.Linear(128, self._z_dim)
        self.eta_encoder = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.variance_encoder = nn.Linear(128, self._z_dim * 2)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self._z_dim, 128, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    @property
    def z_dim(self):
        return self._z_dim

    @pnn.pyro_method
    def model(self, imgs=None):
        if imgs is None:
            imgs = torch.zeros(1, self._data_dim)

        with pyro.plate('imgs', len(imgs)):
            omega_dist = dist.LKJCorrCholesky(self.z_dim, imgs.new_ones(1))
            omega = pyro.sample('omega', omega_dist.expand([len(imgs)]))
            variance_dist = dist.LogNormal(imgs.new_zeros(self.z_dim),
                                           imgs.new_ones(self.z_dim))
            variances = pyro.sample('variances', variance_dist.to_event(1))
            scales = [torch.diag(torch.sqrt(variances[i])) @ omega[i] for i
                      in range(len(imgs))]
            scale_tril = torch.stack(scales, dim=0)

            mu = imgs.new_zeros(self.z_dim)
            zs_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
            zs = pyro.sample('z', zs_dist.to_event(1)).view(-1, self.z_dim, 1,
                                                            1)

            reconstruction = torch.sigmoid(self.decode(zs))
            reconstruction_dist = dist.Bernoulli(reconstruction).to_event(3)
            pyro.sample('reconstruction', reconstruction_dist, obs=imgs)
        return mu, scale_tril, zs, reconstruction

    @pnn.pyro_method
    def guide(self, imgs):
        features = self.encoder_convs(imgs).view(len(imgs), -1)
        features = self.encoder_linears(features)

        eta = self.eta_encoder(features).mean(dim=0)
        omega_dist = dist.LKJCorrCholesky(self.z_dim, eta).expand([len(imgs)])
        omega = pyro.sample('omega', omega_dist)

        vars_params = self.variance_encoder(features).view(-1, 2, self.z_dim)
        variance_dist = dist.LogNormal(vars_params[:, 0],
                                       F.softplus(vars_params[:, 1]))
        variances = pyro.sample('variances', variance_dist.to_event(1))
        scales = [torch.diag(torch.sqrt(variances[i])) @ omega[i] for i
                  in range(len(imgs))]
        scale_tril = torch.stack(scales, dim=0)

        mu = self.mu_encoder(features)
        z_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril).to_event(1)
        return pyro.sample('z', z_dist)

    def forward(self, imgs=None):
        if imgs is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(imgs=imgs)
            return pyro.poutine.replay(self.model, trace=trace)(imgs=imgs)
        return self.model(imgs=None)
