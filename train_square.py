import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")

import shutil
from lightning import seed_everything
import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from models import *
from experiment import VAEXperiment
from thesis_utils.squares_dataset import SquaresDataModule
from torchvision import transforms

from pathlib import Path
import os

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "configs/square_vq_vae.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    # DHA: Added tag to save different runs
    tag = config["data_params"]["tag"] if "tag" in config["data_params"] else ""
    config_tagged = config_path.split("/")[-1].replace(".yaml", f"{tag}.yaml")
    print(config_tagged)

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["logging_params"]["name"] + tag,
    )

    log_path = tb_logger.log_dir
    print(log_path)
    os.makedirs(log_path, exist_ok=True)
    shutil.copy(config_path, log_path + "/" + config_tagged)

    # For reproducibility
    seed_everything(config["exp_params"]["manual_seed"], True)

    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    experiment = VAEXperiment(model, config["exp_params"])

    # Start Training
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        log_every_n_steps=1,
        # strategy=DDPPlugin(find_unused_parameters=False),
        **config["trainer_params"],
    )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {config['model_params']['name']} =======")

    square_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    data_module = SquaresDataModule(
        folder_path=config["data_params"]["root"],
        transform=square_transforms,
        batch_size=config["data_params"]["batch_size"],
        seed=config["exp_params"]["manual_seed"],
    )
    # TODO: Checkpoints?
    runner.fit(experiment, datamodule=data_module)
