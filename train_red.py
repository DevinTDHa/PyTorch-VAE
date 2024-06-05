import shutil
import torch
from pytorch_lightning import LightningDataModule
from torchvision.datasets import VisionDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from experiment import VAEXperiment

# from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import os

if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dim = (3, 32, 32)

    batch_size = 1
    N = 16

    train_shape = [N, *img_dim]
    X = torch.zeros(train_shape).to(device)
    red_values = torch.linspace(0.0, 1.0, N)
    X[:, 0, ...] += red_values.reshape((N, 1, 1)).to(device)

    overfit_batch = (X, torch.Tensor([-1]))  # Label is not used in train_vae_on_batch
    print("Mean/Std R Val", X[:, 0, ...].mean().item(), X[:, 0, ...].std().item())
    print("Mean/Std B Val", X[:, 1, ...].mean().item(), X[:, 1, ...].std().item())
    print("Mean/Std G Val", X[:, 2, ...].mean().item(), X[:, 2, ...].std().item())

    plt.imshow(make_grid(X, nrow=4).permute(1, 2, 0).cpu().detach().numpy())

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
            return self.data[idx], -1

    class RedLT(LightningDataModule):

        def __init__(self, N, img_dim, batch_size):
            super().__init__()
            self.N = N
            self.img_dim = img_dim
            self.batch_size = batch_size

        def setup(self, stage=None):
            self.dataset = RedValuesDataset(self.N, self.img_dim)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

    config_path = "configs/red_vq_vae.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["model_params"]["name"],
    )
    shutil.copy(config_path, tb_logger.log_dir)

    # For reproducibility
    # seed_everything(config["exp_params"]["manual_seed"], True)

    model = vae_models[config["model_params"]["name"]](**config["model_params"])
    experiment = VAEXperiment(model, config["exp_params"])

    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="VQ_Loss",
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
    runner.fit(experiment, datamodule=RedLT(N, img_dim, N))
