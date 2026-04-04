import torch

from src.models.unet import EpsilonUNet


def test_unet_output_shape_matches_input() -> None:
    model = EpsilonUNet(
        in_channels=1,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=1,
        time_embed_dim=128,
        dropout=0.0,
    )

    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(1, 1001, (4,), dtype=torch.long)
    y = model(x, t)

    assert y.shape == x.shape
