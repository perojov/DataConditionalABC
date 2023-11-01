import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class PENDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: torch.Tensor,
        train_params: torch.Tensor,
        val_paths: torch.Tensor,
        val_params: torch.Tensor,
    ):
        super().__init__()
        self.train_paths = train_paths
        self.train_params = train_params
        self.val_paths = val_paths
        self.val_params = val_params

    def setup(self, stage: str):
        self.train = TensorDataset(self.train_paths, self.train_params)
        self.validation = TensorDataset(self.val_paths, self.val_params)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1024, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=256, num_workers=2)


class MarkovExchangeableNeuralNetwork(pl.LightningModule):
    def __init__(self, input_shape: int = 2, nparams: int = 4, hidden_size: int = 100):
        super().__init__()
        self.l1 = torch.nn.Linear(input_shape, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size + 1, hidden_size)
        self.l4 = torch.nn.Linear(hidden_size, nparams)
        self.loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x0 = x[0]
            x = x.unfold(0, 2, 1)
            x = self.l1(x).clamp(min=0)
            x = self.l2(x)
            x = torch.mean(x, 0)
            x = self.l3(torch.cat((x0.reshape(1), x))).clamp(min=0)
            x = self.l4(x)
        else:
            x = x.unfold(1, 2, 1)
            x0 = x[:, 0, 0]
            length = x0.shape[0]
            x0 = torch.reshape(x0, (length, 1))
            x = self.l1(x).clamp(min=0)
            x = self.l2(x)
            x = torch.mean(x, 1)
            x = self.l3(torch.cat((x0, x), 1)).clamp(min=0)
            x = self.l4(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        J = self.loss(output, y)
        self.log("train_loss", J, prog_bar=True)
        return J

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        J = self.loss(output, y)
        self.log("val_loss", J, prog_bar=True)
        return J
