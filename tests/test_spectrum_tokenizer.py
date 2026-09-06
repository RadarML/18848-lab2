"""Tests for the GRT spectrum tokenizer."""

import pytest
import torch
from nrdk.roverd import SpectrumData

from grt.tokenizer import SpectrumTokenizer

D_MODEL = 64
N_CHANNELS = 2
PATCH = (1, 2, 2, 8, 4)  # (time, doppler, elevation, azimuth, range)


def make_spectrum(
    batch: int = 3, shape: tuple[int, ...] = (2, 4, 2, 8, 16)
) -> SpectrumData:
    """Create random spectrum data with the given (t, d, el, az, rng) shape."""
    return SpectrumData(
        spectrum=torch.randn(batch, *shape, N_CHANNELS),
        timestamps=torch.zeros(batch, shape[0], dtype=torch.float64),
        range_resolution=torch.full((batch,), 0.1),
        doppler_resolution=torch.full((batch,), 0.05))


def test_output_shape() -> None:
    """One token per patch, plus the readout token."""
    tokenizer = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, n_channels=N_CHANNELS)

    tokens = tokenizer(make_spectrum(batch=3))

    # (2, 4, 2, 8, 16) / (1, 2, 2, 8, 4) = (2, 2, 1, 1, 4) -> 16 patches.
    assert tokens.shape == (3, 16 + 1, D_MODEL)


def test_squeeze_folds_axes_into_channels() -> None:
    """Squeezed axes are moved to the channel axis instead of being patched."""
    tokenizer = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, squeeze=[2, 3], n_channels=N_CHANNELS)

    tokens = tokenizer(make_spectrum(batch=3))

    # Elevation and azimuth are consumed by the squeeze, leaving
    # (2, 4, 16) / (1, 2, 4) = (2, 2, 4) -> 16 patches, unchanged here since
    # both squeezed axes were already a single patch wide.
    assert tokens.shape == (3, 16 + 1, D_MODEL)
    assert tokenizer.squeeze is not None
    assert tokenizer.squeeze.n_channels == PATCH[2] * PATCH[3]


def test_readout_token_is_appended_last() -> None:
    """The last token of every sequence is the (shared) readout parameter."""
    tokenizer = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, n_channels=N_CHANNELS)

    tokens = tokenizer(make_spectrum(batch=3))

    assert torch.equal(
        tokens[:, -1], tokenizer.readout.readout.expand(3, D_MODEL))


def test_positional_embedding_distinguishes_patches() -> None:
    """Identical patches at different positions get different tokens."""
    tokenizer = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, n_channels=N_CHANNELS)

    # Two range patches with exactly the same contents.
    block = torch.randn(1, 1, 2, 2, 8, 4, N_CHANNELS)
    spectrum = make_spectrum(batch=1, shape=(1, 2, 2, 8, 8))
    spectrum.spectrum = torch.cat([block, block], dim=5)

    tokens = tokenizer(spectrum)

    assert tokens.shape == (1, 2 + 1, D_MODEL)
    assert not torch.allclose(tokens[0, 0], tokens[0, 1])


def test_rejects_wrong_patch_rank() -> None:
    """The patch size must cover all five spectrum axes."""
    with pytest.raises(ValueError, match="expected 5 dims"):
        SpectrumTokenizer(patch=(2, 2, 8, 4))
