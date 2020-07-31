import numpy as np
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.distributions.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ContinuousBernoulli(torch.distributions.ContinuousBernoulli,
                          dist.torch_distribution.TorchDistributionMixin):
    pass

class Wishart(dist.TorchDistribution):
    has_rsample = True

    def __init__(self, df, scale, validate_args=None):
        self._dim = scale.shape[-1]
        assert df > self._dim - 1
        self.df = df
        self.cholesky_factor = transforms.LowerCholeskyTransform()(scale)
        self.chi_sqd_dists = [dist.Chi2(self.df - i) for i in range(self._dim)]
        batch_shape, event_shape = scale.shape[:-2], scale.shape[-2:]
        super().__init__(batch_shape, event_shape, validate_args)

    def rsample(self, sample_shape=torch.Size()):
        A = torch.eye(self._dim).expand(sample_shape, self._dim, self._dim)
        A = A.to(self.cholesky_factor)
        chi_sqds = torch.stack([d.rsample(sample_shape)
                                for d in self._chi_sqd_dists], dim=-1)
        A = torch.tril(torch.randn(*sample_shape, *A.shape), diagonal=-1) +\
            A * chi_sqds

        return self.cholesky_factor @ (A @ A.t()) @ self.cholesky_factor.t()

    def log_prob(self, value):
        scale = self.cholesky_factor @ self.cholesky_factor.t()
        log_normalizer = (self.df * self._dim / 2) * np.log(2) +\
                         (self.df / 2) * torch.logdet(scale) +\
                         torch.mvlgamma(self.df / 2, self._dim)

        numerator_logdet = ((self.df - self._dim - 1) / 2) * torch.logdet(value)
        numerator_logtrace = -1/2 * torch.trace(
            torch.cholesky_inverse(self.cholesky_factor) @ value
        )
        log_numerator = numerator_logdet + numerator_logtrace
        return log_numerator - log_normalizer

class InverseWishart(dist.TorchDistribution):
    has_rsample = True

    def __init__(self, df, scale, validate_args=None):
        self.base_wishart = Wishart(df, torch.inverse(scale), validate_args)

    def rsample(self, sample_shape=torch.Size()):
        return torch.inverse(self.base_wishart.rsample(sample_shape))

    def log_prob(self, value):
        return self.base_wishart.log_prob(torch.inverse(value))

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
        self.scale_encoder = nn.Linear(128, self._z_dim ** 2)
        self.variance_encoder = nn.Linear(self._z_dim, self._z_dim * 2)

        self.decoder_linears = nn.Sequential(
            nn.Linear(self._z_dim, 128), nn.ReLU(),
            nn.Linear(128, 4 * 4 * 64), nn.ReLU(),
        )
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

        self.lower_choleskyize = transforms.LowerCholeskyTransform()

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
            scale_tril = self.lower_choleskyize(torch.stack(scales, dim=0))

            mu = imgs.new_zeros(self.z_dim)
            zs_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
            zs = pyro.sample('z', zs_dist)

            features = self.decoder_linears(zs).view(-1, 64, 4, 4)
            reconstruction = self.decoder_convs(features)
            reconstruction_dist = ContinuousBernoulli(logits=reconstruction)
            pyro.sample('reconstruction', reconstruction_dist.to_event(3),
                        obs=imgs)
        return mu, scale_tril, zs, reconstruction

    @pnn.pyro_method
    def guide(self, imgs):
        features = self.encoder_convs(imgs).view(len(imgs), -1)
        features = self.encoder_linears(features)

        with pyro.plate('imgs', len(imgs)):
            mu = self.mu_encoder(features)
            scale_tril = self.scale_encoder(features).view(-1, self.z_dim,
                                                           self.z_dim)
            scale_tril = self.lower_choleskyize(scale_tril)
            z_dist = dist.MultivariateNormal(mu, scale_tril=scale_tril)
            zs = pyro.sample('z', z_dist)

            omega_dist = dist.LKJCorrCholesky(self.z_dim, imgs.new_ones(1)).expand([len(imgs)])
            omega = pyro.sample('omega', omega_dist)

            vars_params = self.variance_encoder(zs).view(-1, 2, self.z_dim)
            variance_dist = dist.LogNormal(vars_params[:, 0],
                                           F.softplus(vars_params[:, 1]))
            variances = pyro.sample('variances', variance_dist.to_event(1))
            scales = [torch.diag(torch.sqrt(variances[i])) @ omega[i] for i
                      in range(len(imgs))]
            scale_tril = self.lower_choleskyize(torch.stack(scales, dim=0))

            return zs, scale_tril

    def forward(self, imgs=None):
        if imgs is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(imgs=imgs)
            return pyro.poutine.replay(self.model, trace=trace)(imgs=imgs)
        return self.model(imgs=None)
