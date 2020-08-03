import collections
import numpy as np
import probtorch
import torch
import torch.distributions as dist
import torch.distributions.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Wishart(dist.Distribution):
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
        A = torch.tril(torch.randn(*sample_shape, self._dim, self._dim,
                                   device=self.df.device),
                       diagonal=-1) + chi_sqds

        results = []
        chol_a_mat = torch.bmm(self.cholesky_factor, a_mat)
        return torch.bmm(chol_a_mat, chol_a_mat.transpose(-2, -1))

    def log_prob(self, value):
        chol = self.cholesky_factor

        scale = torch.bmm(chol, chol.transpose(-2, -1))
        log_normalizer = (self.df * self._dim / 2.) * np.log(2) +\
                         (self.df / 2.) * torch.logdet(scale) +\
                         torch.mvlgamma(self.df / 2., self._dim)

        numerator_logdet = (self.df - self._dim - 1) / 2. * torch.logdet(value)
        choleskied_value = torch.bmm(torch.inverse(chol), value)
        numerator_logtrace = -1/2 * torch.diagonal(choleskied_value, dim1=-2, dim2=-1).sum(-1)
        log_numerator = numerator_logdet + numerator_logtrace
        return log_numerator - log_normalizer

class InverseWishart(dist.Distribution):
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

        self.register_buffer('cov_loc', torch.eye(self.z_dim))
        self.register_buffer('cov_df', torch.tensor([self.z_dim + 1]))

    @property
    def z_dim(self):
        return self._z_dim

    def model(self, p, q=None, imgs=None):
        if not q:
            q = collections.defaultdict(lambda x: None)
        if imgs is None:
            imgs = torch.zeros(1, np.sqrt(self._data_dim),
                               np.sqrt(self._data_dim))

        cov_loc = self.cov_loc.expand(imgs.shape[0], self.z_dim, self.z_dim)
        covariance = p.variable(InverseWishart, self.cov_df, cov_loc,
                                name='covariance', value=q['covariance'].value)

        mu = imgs.new_zeros(self.z_dim)
        zs = p.multivariate_normal(loc=mu, covariance_matrix=covariance,
                                   name='z', value=q['z'].value)

        features = self.decoder_linears(zs).view(-1, 64, 4, 4)
        reconstruction = torch.sigmoid(self.decoder_convs(features))
        reconstruction = p.continuous_bernoulli(probs=reconstruction,
                                                name='reconstruction',
                                                value=imgs)

        return mu, covariance, zs, reconstruction

    def guide(self, q, imgs):
        features = self.encoder_convs(imgs).view(len(imgs), -1)
        features = self.encoder_linears(features)

        mu = self.mu_encoder(features)
        A = self.scale_encoder(features).view(-1, self.z_dim, self.z_dim)
        L = torch.tril(A)
        diagonal = F.softplus(A.diagonal(0, -2, -1)) + 1e-4
        L = L + torch.diag_embed(diagonal)
        L_LT = torch.bmm(L, L.transpose(-2, -1))
        covariance = L_LT + 1e-4 * torch.eye(self.z_dim, device=imgs.device)
        zs = q.multivariate_normal(loc=mu, covariance_matrix=covariance,
                                   name='z')

        cov_loc = self.cov_loc.expand(imgs.shape[0], self.z_dim, self.z_dim)
        zs_squared = torch.bmm(zs.unsqueeze(-1), zs.unsqueeze(-2))
        q.variable(InverseWishart, self.cov_df + 1, cov_loc + zs_squared,
                   name='covariance', value=covariance)

        return zs, covariance

    def forward(self, imgs=None):
        q = probtorch.Trace()
        self.guide(q, imgs)

        p = probtorch.Trace()
        _, _, _, reconstruction = self.model(p, q, imgs)

        return p, q, reconstruction
