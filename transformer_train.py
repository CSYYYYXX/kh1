import os
import time
import argparse

import numpy as np
import paddle
import yaml

from paddle import to_tensor
from paddle.io import DataLoader
from paddle.optimizer import Adam
import paddle.fluid as fluid
from paddle.nn import MSELoss
from paddle.fluid.dygraph import to_variable

import matplotlib.pyplot as plt

from src.dataset import create_transformer_dataset
from src import plot_train_loss
from src.model_transformer import Informer
from cae_prediction import cae_prediction


np.random.seed(42)
paddle.seed(42)


def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



class CustomWithLossCell(fluid.dygraph.Layer):
    def __init__(self, backbone, loss_fn, args):
        super(CustomWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.args = args

    def forward(self, seq_x, seq_y):

        batch_x = seq_x
        batch_y = seq_y

        if self.args["padding"] == 0:
            dec_inp = paddle.zeros(
                (batch_y.shape[0], self.args["pred_len"], batch_y.shape[-1]), dtype='float32'
            )
        else:
            dec_inp = paddle.ones(
                (batch_y.shape[0], self.args["pred_len"], batch_y.shape[-1]), dtype='float32'
            )
        dec_inp = paddle.concat([batch_x[:, - self.args["label_len"]:, :], dec_inp], axis=1)

        outputs = self._backbone(batch_x, dec_inp)
        batch_y = batch_y[:, -self.args["pred_len"]:, :]

        return self._loss_fn(outputs, batch_y)


def train():
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["transformer_data"]
    model_params = config["transformer"]
    optimizer_params = config["transformer_optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    ckpt_dir = os.path.join(summary_dir, "ckpt")

    # prepare model
    with fluid.dygraph.guard():
        model = Informer(**model_params)
        loss_fn = paddle.nn.MSELoss()
        optimizer = Adam(
            learning_rate=optimizer_params["lr"],
            parameters=model.parameters(),
            weight_decay=optimizer_params["weight_decay"],
        )

        # prepare dataset
        latent_true = cae_prediction(args.config_file_path)
        dataloader, _ = create_transformer_dataset(
            latent_true,
            data_params["batch_size"],
            data_params["time_size"],
            data_params["latent_size"],
            data_params["time_window"],
            data_params["gaussian_filter_sigma"],
        )

        time_now = time.time()
        loss_net = CustomWithLossCell(model, loss_fn, data_params)

        for epoch in range(optimizer_params["epochs"]):
            print(f">>>>>>>>>>>>>>>>>>>>Train_{epoch}<<<<<<<<<<<<<<<<<<<<")
            ts = time.time()

            for batch_id, data in enumerate(dataloader):
                seq_x, seq_y = data
                seq_x = to_variable(seq_x)
                seq_y = to_variable(seq_y)

                # 打印每个批次的数据形状
                print(f"Batch ID: {batch_id}")
                print(f"seq_x shape: {seq_x.shape}")
                print(f"seq_y shape: {seq_y.shape}")

                loss = loss_net(seq_x, seq_y)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                if batch_id % 50 == 0:
                    print(f"Epoch: {epoch}, batch: {batch_id}, loss is: {loss.numpy()}")

            paddle.save(model.state_dict(), os.path.join(ckpt_dir, f"Informer_{epoch}.ckpt"))

            print("Train Time Cost:", time.time() - ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transformer net for KH")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    print(f"pid: {os.getpid()}")
    train()
