import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from src.config import Config, ExpConfig

import wandb

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))  # exp/ex01
project_root = os.path.join(current_dir, "../../")  # rootへ移動
sys.path.append(os.path.normpath(project_root))

# ruffの警告を無視
from utils.env import EnvConfig  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.timing import trace  # noqa: E402

load_dotenv()
LOGGER = None


# hydra用にdefaultを設定
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


####################
# 実験用コード
####################
def log_config(cfg: Config, LOGGER) -> None:
    LOGGER.info(
        "\nConfig: %s",
        json.dumps(OmegaConf.to_container(cfg, resolve=True), default=str, indent=4),
    )


def init_output_dir(cfg: Config, exp_name: str):
    output_dir = Path(cfg.env.output_dir) / exp_name
    cfg.env.exp_output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(cfg: Config, LOGGER) -> None:
    """設定をexp_output_dirにconfig.yamlとして保存する"""
    config_path = Path(cfg.env.exp_output_dir) / "config.yaml"
    OmegaConf.save(cfg.exp, config_path)
    LOGGER.info(f"Config saved to: {config_path}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(
    cfg: Config,
) -> None:  # Duck typing: cfgは実際にはDictConfigだが、Configクラスのように扱える
    global LOGGER
    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices['exp']}"  # e.g. 000_sample/default
    exp_name += f"_{cfg.exp.name}" if cfg.exp.name != "" else ""
    output_dir = init_output_dir(cfg, exp_name)
    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("output_dir: %s", output_dir)
    LOGGER.info("Start")

    with trace("sleep"):
        time.sleep(1.1)

    log_config(cfg, LOGGER)
    save_config(cfg, LOGGER)

    wandb.init(  # type: ignore[attr-defined]
        project=cfg.exp.wandb_project_name,
        name=exp_name,
        notes=", ".join(HydraConfig.get().overrides.task),  # オーバーライドの内容
        config=cast(dict[str, Any], OmegaConf.to_container(cfg.exp, resolve=True)),
        mode="disabled" if cfg.exp.debug else "online",
    )


if __name__ == "__main__":
    main()
