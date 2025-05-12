import hydra
from omegaconf import DictConfig

from train import SleepStageTrainer, set_seed


@hydra.main(config_path="../configs", config_name="config_hydra", version_base=None)
def main(cfg: DictConfig):
    """
    Main function wrapped with Hydra for config management for training and validation.

    Args:
        cfg (DictConfig): Configuration object loaded from YAML file.
    """
    set_seed(cfg.general.seed)
    trainer = SleepStageTrainer(cfg)
    
    if cfg.run_mode == "train":
        print("Training process...")
        trainer.training()
    elif cfg.run_mode == "validate":
        print("Validation process...")
        trainer.validate()
    else:
        trainer.training()
        trainer.validate()


if __name__ == "__main__":
    main()