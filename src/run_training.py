import hydra
from omegaconf import DictConfig
import torch


from train import run_training 


@hydra.main(config_path="../configs", config_name="config_hydra", version_base=None)
def main(cfg: DictConfig):
    """
    Main function wrapped with Hydra for config management for training.

    Args:
        cfg (DictConfig): Configuration object loaded from YAML file.
    """

    # Run training with Hydra config
    run_training(cfg)


if __name__ == "__main__":
    main()
