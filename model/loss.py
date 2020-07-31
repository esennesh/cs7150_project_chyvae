import probtorch.objectives.montecarlo as objectives

def elbo(output, target):
    p, q = output

    return -objectives.elbo(q, p, sample_dim=None, batch_dim=0)
