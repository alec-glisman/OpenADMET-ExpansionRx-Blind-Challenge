import torch
from admet.model.chemprop.model import ChempropLightning


def test_chemprop_lightning_forward():
    model = ChempropLightning(
        model_params={
            "atom_fdim": 10,
            "bond_fdim": 5,
            "depth": 2,
            "hidden_size": 16,
            "dropout": 0.1,
        },
        lr=1e-3,
        weight_decay=1e-5,
        target_dim=2,
        task_weights=[0.5, 0.5],
    )

    x = torch.randn(4, 10)  # NOTE: this is a rough stand-in, real chemprop uses graph features
    y_hat = model(x)
    assert y_hat.shape == (4, 2)
