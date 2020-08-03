from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
import wget

from base import BaseDataLoader

DSPRITES_URL = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
DSPRITE_FILENAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

class dSprites(Dataset):
    def __init__(self, root='./', download=True, transform=None, include_discrete=True):
        self.root = root
        self.transform = transform
        self.n_factors = 5 if include_discrete else 4
        self.diff = 1 if include_discrete else 2

        if not os.path.exists(self.root + DSPRITE_FILENAME) and download:
            wget.download(DSPRITES_URL, self.root + DSPRITE_FILENAME)

        # Load dataset
        dataset_zip = np.load(self.root + DSPRITE_FILENAME, encoding='latin1', allow_pickle=True)
        print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ]))
        )
        #print('Metadata: \n', metadata)
        print('Dataset loaded : OK.')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.fromarray(self.imgs[idx])
        latent = self.latents_values[idx]

        if self.transform is not None:
            image = self.transform(image)
            return (image, latent)

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def sample_fixed_factor(self, size=20):
        if self.n_factors == 5:
            factor = np.random.randint(5) + 1
        else:
            factor = np.random.randint(4) + 2
        latents_sampled = self.sample_latent(size=size)
        latents_sampled[:, factor] = latents_sampled[0, factor]
        idx = self.latent_to_index(latents_sampled)
        xi = self.imgs[idx].reshape(size, 64, 64, 1)
        return xi, factor

class dSpriteDataLoader(BaseDataLoader):
    """
    dSprites data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = dSprites(self.data_dir, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
