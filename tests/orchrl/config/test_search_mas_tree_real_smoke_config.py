from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import yaml


def test_search_mas_tree_real_smoke_config_has_minimal_real_smoke_settings() -> None:
    config_dir = Path("orchrl/config/search").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="search_mas_tree_real_smoke")

    assert cfg.training.total_training_steps == 2
    assert cfg.training.val_freq == 1
    assert cfg.training.train_batch_size == 1
    assert cfg.training.train_sample_num == 1
    assert cfg.training.if_save is False

    assert cfg.training.mate.rollout_mode == "tree"
    assert cfg.training.mate.tree.k_branches == 1
    assert cfg.training.mate.prompt_loader.source_type == "jsonl"

    prompt_path = Path(cfg.training.mate.prompt_loader.path)
    assert prompt_path.exists(), str(prompt_path)

    raw_cfg = yaml.safe_load(Path("orchrl/config/search/search_mas_tree_real_smoke.yaml").read_text())
    for model_key in ("model_0", "model_1", "model_2"):
        model_cfg = raw_cfg["models"][model_key]["ppo_trainer_config"]
        actor_model_cfg = model_cfg["actor_rollout_ref"]["model"]
        rollout_cfg = model_cfg["actor_rollout_ref"]["rollout"]
        critic_model_cfg = model_cfg["critic"]["model"]

        assert actor_model_cfg["use_remove_padding"] is False
        assert critic_model_cfg["use_remove_padding"] is False
        assert actor_model_cfg["override_config"]["_attn_implementation"] == "eager"
        assert actor_model_cfg["override_config"]["_attn_implementation_internal"] == "eager"
        assert rollout_cfg["gpu_memory_utilization"] == 0.92
        assert rollout_cfg["enforce_eager"] is True
        assert critic_model_cfg["override_config"]["_attn_implementation"] == "eager"
        assert critic_model_cfg["override_config"]["_attn_implementation_internal"] == "eager"
