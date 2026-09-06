"""GRT spectrum linear patch tokenizer."""

from collections.abc import Sequence

from jaxtyping import Float
from nrdk import modules
from nrdk.roverd import SpectrumData
from torch import Tensor, nn


class SpectrumTokenizer(nn.Module):
    """GRT 4D Radar Spectrum tokenizer.

    Positional embeddings are n-dimensional, splitting the input features into
    `d` equal chunks encoding each axis separately.

    !!! info

        We use a relative coordinate system for positional embeddings instead
        of absolute position indices, where each axis is scaled to `[-1, 1]` by
        default and scaled by `pos_scale` and `global_scale` factors; see
        [`modules.Sinusoid`][nrdk.] for details.

    !!! warning

        Any axes specified in `squeeze` must have a patch size which is equal
        to the input size along that axis.

    Args:
        d_model: model feature dimension.
        patch: input (doppler, azimuth, elevation, range) patch size.
        squeeze: eliminate these axes by moving them to the channel axis prior
            to patching; specified by index.
        n_channels: number of input channels; see [`xwr.nn`][xwr.nn].
        scale: position embedding scale.
        w_min: minimum frequency for sinusoidal position embeddings.
    """

    def __init__(
        self, d_model: int = 768, patch: Sequence[int] = (1, 2, 2, 8, 4),
        squeeze: Sequence[int] = [], n_channels: int = 2,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
    ) -> None:
        super().__init__()

        if len(patch) != 5:
            raise ValueError(
                f"Invalid patch size: {patch}; expected 5 dims "
                f"(time, doppler, elevation, azimuth, range)")

        if len(squeeze) > 0:
            self.squeeze = modules.Squeeze(dim=squeeze, size=patch)
            n_channels = n_channels * self.squeeze.n_channels
            patch = [p for i, p in enumerate(patch) if i not in squeeze]
        else:
            self.squeeze = None

        self.patch = modules.PatchMerge(
            d_in=n_channels, d_out=d_model, scale=patch, norm=False)

        self.pos = modules.Sinusoid(scale=scale, w_min=w_min)
        self.readout = modules.Readout(d_model=d_model)

    def forward(
        self, spectrum: SpectrumData
    ) -> Float[Tensor, "n s c"]:
        """Apply radar transformer.

        Args:
            spectrum: input batch spectrum data.

        Returns:
            Tokenized output.
        """
        x = spectrum.spectrum

        if self.squeeze is not None:
            x = self.squeeze(x)

        embedded = self.pos(self.patch(x))
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])

        return self.readout(flat)
