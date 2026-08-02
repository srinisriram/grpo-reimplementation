import pytest


@pytest.mark.integration
def test_config_imports_without_model_download():
    from grpo.config import load_config

    config = load_config("configs/smoke.yaml")
    assert config.group_size == 2
    assert config.model_name.startswith("Qwen/")

