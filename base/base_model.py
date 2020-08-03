import probtorch
import torch
import numpy as np
from abc import abstractmethod

EPS = 1e-9

class BaseModel(torch.nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])

def probtorch_cross_entropy(estimate, ground_truth):
    terms = -(torch.log(estimate + EPS) * ground_truth +\
              torch.log(1 - estimate + EPS) * (1 - ground_truth))
    return terms.squeeze().sum(-1).sum(-1)
