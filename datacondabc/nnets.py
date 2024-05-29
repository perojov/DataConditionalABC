import torch
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class PENDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths: torch.tensor,
        train_params: torch.tensor,
        val_paths: torch.tensor,
        val_params: torch.tensor,
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
        return DataLoader(
            self.train,
            batch_size=512,
            num_workers=5,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=128,
            num_workers=5,
            pin_memory=True,
            persistent_workers=True,
        )


class PENCNN(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        output_shape,
        pen_nr=3,
        con_layers=[25, 50, 100],
        dense_layers=[100, 100, 100],
    ):
        super().__init__()
        self.pen_nr = pen_nr
        self.input_shape = input_shape
        self.con_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.bnorm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.loss = torch.nn.MSELoss()
        d = input_shape[1]

        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1],
            out_channels=con_layers[0],
            kernel_size=pen_nr + 1,
            stride=1,
            padding=0,
        )

        in_channels = con_layers[0]
        for out_channels in con_layers[1:]:
            self.con_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            in_channels = out_channels

        self.avg_pool = nn.AvgPool1d(
            kernel_size=input_shape[0] - pen_nr, stride=1, padding=0
        )

        # Dense layers with  ReLU activation.
        in_features = con_layers[-1] + (pen_nr + 1) * d
        for out_features in dense_layers:
            self.dense_layers.append(
                nn.Linear(in_features=in_features, out_features=out_features)
            )
            # self.dropout_layers.append(nn.Dropout(p=0.2))
            # self.bnorm_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        self.dropout_layers.append(nn.Dropout(p=0.2))

        # Output layer
        self.output_layer = nn.Linear(
            in_features=dense_layers[-1], out_features=output_shape
        )

    def forward(self, x):
        original_x = x

        # Convolution, pool, flatten.
        x = self.conv1(x).clamp(min=0)
        for conv_layer in self.con_layers:
            x = conv_layer(x).clamp(min=0)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        # PEN magick.
        transposed_x = original_x.transpose(1, 2)
        cut_input = transposed_x[:, 0 : self.pen_nr + 1, :]
        cut_input_1d = cut_input.reshape(cut_input.shape[0], -1)
        x = torch.cat((x, cut_input_1d), dim=1)

        # Pass through Dense layers with ReLU activation.
        for dense_layer in self.dense_layers:
            x = dense_layer(x).clamp(min=0)

        # Dropout
        x = self.dropout_layers[0](x)

        # Output layer
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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
