# tests/test_core.py
import pytest
import torch
from src.global_upreg_seq.core import generate_simulated_data


@pytest.mark.parametrize("dist_type", [
    "poisson",
    "negative_binomial",
    "zinb",
    "zero_inflated_poisson"
])
def test_generate_simulated_data_distributions(dist_type):
    counts, labels = generate_simulated_data(
        5,
        gene_size=100,
        distribution_type=dist_type,
    )
    
    assert counts.shape == (10, 100)
    assert labels.shape == (10,)
    assert torch.all(labels[:5] == 0)
    assert torch.all(labels[5:] == 1)


