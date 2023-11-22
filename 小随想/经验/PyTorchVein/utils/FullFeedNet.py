from einops import rearrange
from torch import Tensor, nn


class FFN(nn.Module):
    def __init__(self, data_size: int = 240, layer_size: int = 3) -> None:
        super().__init__()
        self.data_size = data_size
        self.channels = data_size**2
        ffn_layers = []
        for _ in range(layer_size):
            ffn_layers.append(nn.Linear(self.channels, self.channels))
        self.ffn_layers = nn.Sequential(*ffn_layers)
        self.to_out = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        assert len(inputs.shape) == 4
        b, c, h, w = inputs.shape
        inputs = rearrange(inputs, "b c h w -> b c (h w)", b=b, c=c, h=h, w=w)
        outputs = self.ffn_layers(inputs)
        outputs = self.to_out(outputs)
        outputs = rearrange(outputs, "b c hw -> b c h w", b=b, c=c, h=h, w=w)

        return outputs
