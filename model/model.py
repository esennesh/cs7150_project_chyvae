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
        sample_shape = self.batch_shape + sample_shape

        chi_sqds = torch.stack([d.rsample(sample_shape)
                                for d in self.chi_sqd_dists], dim=-1)
        chi_sqds = torch.stack([torch.diag(chi_sqd) for chi_sqd
                                in torch.unbind(chi_sqds, dim=0)], dim=0)
        A_tril = torch.tril(torch.randn(*sample_shape, self._dim, self._dim),
                            diagonal=-1).to(self.cholesky_factor)
        A = chi_sqds.to(self.cholesky_factor) + A_tril

        results = []
        for chol, a_mat in zip(torch.unbind(self.cholesky_factor, dim=0),
                               torch.unbind(A, dim=0)):
            results.append(chol @ (a_mat @ a_mat.t()) @ chol.t())
        return torch.stack(results, dim=0)

    def log_prob(self, value):
        cholesky_factor = self.cholesky_factor.to(value)

        scale = torch.stack([chol @ chol.t() for chol in
                             torch.unbind(cholesky_factor, dim=0)], dim=0)
        df_factor = torch.tensor([self.df / 2]).to(value)
        log_normalizer = (self.df * self._dim / 2) * np.log(2) +\
                         (self.df / 2) * torch.logdet(scale) +\
                         torch.mvlgamma(df_factor, self._dim)

        numerator_logdet = (self.df - self._dim - 1) / 2 * torch.logdet(value)
        choleskied_value = torch.stack([
            torch.trace(torch.cholesky_inverse(cholesky_factor[i]) @ value[i])
            for i in range(value.shape[0])
        ], dim=0)
        numerator_logtrace = -1/2 * choleskied_value
        log_numerator = numerator_logdet + numerator_logtrace
        return log_numerator - log_normalizer

class InverseWishart(dist.TorchDistribution):
    has_rsample = True

    def __init__(self, df, scale, validate_args=None):
        self.base_wishart = Wishart(df, torch.inverse(scale), validate_args)

    @property
    def batch_shape(self):
        return self.base_wishart.batch_shape

    @property
    def event_shape(self):
        return self.base_wishart.event_shape

    def rsample(self, sample_shape=torch.Size()):
        return torch.inverse(self.base_wishart.rsample(sample_shape))

    def log_prob(self, value):
        return self.base_wishart.log_prob(torch.inverse(value))

class ShapesChyVae(BaseModel):
    def __init__(self, z_dim=10, data_dim=64*64):
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
            imgs = torch.zeros(1, np.sqrt(self._data_dim),
                               np.sqrt(self._data_dim))

        with pyro.plate('imgs', len(imgs)):
            eye = torch.eye(self.z_dim).expand(imgs.shape[0], self.z_dim,
                                               self.z_dim)
            inv_wishart = InverseWishart(self.z_dim + 1, eye, False)
            covariance = pyro.sample('covariance', inv_wishart)
            scale_tril = self.lower_choleskyize(covariance)

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

            cov_loc = torch.eye(self.z_dim).expand(imgs.shape[0], self.z_dim,
                                                   self.z_dim).to(imgs)
            zs_squared = torch.stack([z.unsqueeze(-1) @ z.unsqueeze(0) for z
                                      in torch.unbind(zs, dim=0)], dim=0)
            cov_loc = cov_loc + zs_squared
            inv_wishart = InverseWishart(self.z_dim + 2, cov_loc, False)
            covariance = pyro.sample('covariance', inv_wishart)
            scale_tril = self.lower_choleskyize(covariance)

            return zs, scale_tril

    def forward(self, imgs=None):
        if imgs is not None:
            trace = pyro.poutine.trace(self.guide).get_trace(imgs=imgs)
            return pyro.poutine.replay(self.model, trace=trace)(imgs=imgs)
        return self.model(imgs=None)
