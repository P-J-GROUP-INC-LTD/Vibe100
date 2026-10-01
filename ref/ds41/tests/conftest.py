"""Session fixtures: one tiny official model (building it is cheap, but the reference keeps module-level state)."""
import numpy as np
import pytest
import torch

from ref.ds41.tests import official as O


@pytest.fixture(scope="session")
def tiny_cfg():
    return O.tiny_config()


@pytest.fixture(scope="session")
def tiny(tiny_cfg):
    """(official model, tokenizer, DictWeights exported from it, token_map ndarray)."""
    model, tok = O.build_tiny(tiny_cfg, seed=0)
    w = O.export(model)
    tm = model.engram_hash.token_map.numpy().copy()
    return model, tok, w, tm


@pytest.fixture(scope="session")
def official_mod():
    mm, em = O.load_official()
    return mm, em


@pytest.fixture(autouse=True, scope="session")
def _no_grad():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    yield
