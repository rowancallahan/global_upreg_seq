# tests/test_core.py
import pytest
import torch
from src.global_upreg_seq.core import generate_simulated_data, train
from src.global_upreg_seq.models import factor_model_poisson 


@pytest.mark.parametrize("dist_type", [
    "poisson",
    "negative_binomial",
    "zinb",
    "zero_inflated_poisson"
])
def test_generate_simulated_data_distributions(dist_type):
    counts, labels, _ = generate_simulated_data(
        5,
        gene_size=100,
        distribution_type=dist_type,
    )
    
    assert counts.shape == (10, 100)
    assert labels.shape == (10,)
    assert torch.all(labels[:5] == 0)
    assert torch.all(labels[5:] == 1)

@pytest.mark.parametrize("dist_type", [
    "poisson",
    "negative_binomial",
    "zinb",
    "zero_inflated_poisson"
])
def test_full_pipeline(dist_type, group_size=15, training_amount=50):
    counts, labels, _ = generate_simulated_data(group_size,
                                             gene_size=100,
                                             distribution_type=dist_type)
                        

    model = factor_model_poisson() 
    model, guide, (losses, logp) = train(counts,labels,model,num_iterations=training_amount)
    assert len(losses) == training_amount
    

