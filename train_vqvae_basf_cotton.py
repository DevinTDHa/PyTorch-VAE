import shutil
from lightning import seed_everything
import torch
from pytorch_lightning import LightningDataModule
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from experiment import VAEXperiment
from PIL import Image

from torch.utils.data import DataLoader

# from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path
import os
import pandas as pd


class BASFDataset(VisionDataset):
    """Loads the BASF Dataset with either soybean or cotton images or combined images.

    The expected folder structure is as follows:
    root
    ├── annotation_combined.txt
    ├── annotation_soybean.txt
    ├── annotation_cotton.txt
    ├── imgs
    │   ├── {UUID}.jpg
    │   ├── {DATE}.jpg
    │   ├── ...

    """

    dataset_classes = ["combined", "soybean", "cotton"]

    # Dataset Metadata
    # image_size = [3, 512, 512]

    def __init__(
        self,
        root: str,
        dataset_name: str = "cotton",
        keep_in_memory: bool = True,
    ):
        self.config = config
        self.root: str = root

        img_input_size = 512
        self.transform = transforms.Compose(
            # [transforms.Resize(img_input_size), transforms.ToTensor()]
            [transforms.ToTensor()]
        )

        assert (
            dataset_name in self.dataset_classes
        ), f"Invalid class label in {dataset_name}"

        # self.image_size = image_size

        # Prepare labels
        self.attributes = [0]  # DHA: For output_size
        annotation_file = os.path.join(self.root, f"annotation_{dataset_name}.txt")
        self.annotation_df = pd.read_csv(annotation_file, sep="\t")

        self.keep_in_memory = keep_in_memory
        if keep_in_memory:
            self.data = []
            print("Loading data into memory...")
            for idx in tqdm(range(len(self.annotation_df))):
                img, label = self.load_image(idx)
                self.data.append((img, label))

    def load_image(self, index):
        base_name, label = self.annotation_df.iloc[index]
        # DHA: Assume preprocessed images
        img_path = os.path.join(self.root, "imgs_resize512", base_name)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        label = torch.Tensor([label])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __getitem__(self, index):
        if self.keep_in_memory:
            return self.data[index]
        else:
            return self.load_image(index)

    def __len__(self):
        return len(self.annotation_df)


class BASFLightningDataModule(LightningDataModule):

    def __init__(self, root, batch_size, num_workers):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BASFDataset(self.root)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):  # Only to enable generating samples during validation
        return self.train_dataloader()

    def val_dataloader(self):  # Only to enable generating samples during validation
        return self.train_dataloader()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dim = (3, 512, 512)

    config_path = "configs/basf_cotton.yaml"
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    tag = ""  # DHA: Added tag to save different runs

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
        **config["trainer_params"],
    )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {config['model_params']['name']} =======")

    runner.fit(
        experiment,
        datamodule=BASFLightningDataModule(
            root=config["data_params"]["root"],
            batch_size=config["data_params"]["batch_size"],
            num_workers=config["data_params"]["num_workers"],
        ),
    )
