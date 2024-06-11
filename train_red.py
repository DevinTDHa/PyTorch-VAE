import shutil
import torch
from pytorch_lightning import LightningDataModule
from torchvision.datasets import VisionDataset
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from experiment import VAEXperiment

# from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import os


class RedValuesDataset(VisionDataset):
    def __init__(self, N, img_dim):
        super(RedValuesDataset, self).__init__()
        self.N = N
        self.img_dim = img_dim
        self.data = torch.zeros([N, *img_dim])
        self.data[:, 0, ...] += torch.linspace(0.0, 1.0, N).reshape((N, 1, 1))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        random_noise = torch.randn_like(self.data[idx]) * 0.1  # Noise scale
        return self.data[idx] + random_noise, -1


class RedLT(LightningDataModule):

    def __init__(self, N, img_dim, batch_size):
        super().__init__()
        self.N = N
        self.img_dim = img_dim
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = RedValuesDataset(self.N, self.img_dim)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.N)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.N)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dim = (3, 32, 32)
    N = 128

    config_path = "configs/red_vq_vae_defaultb.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    tag = "_defaultb_random_noise"
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
    # seed_everything(config["exp_params"]["manual_seed"], True)

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
    img_dim = (3, 32, 32)

    batch_size = 16
    N = 128
    runner.fit(
        experiment, datamodule=RedLT(N=N, img_dim=img_dim, batch_size=batch_size)
    )
