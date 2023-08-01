import os
import argparse
import numpy as np
import paddle
import yaml
from paddle.io import DataLoader
from paddle.static import load_program_state

from src.postprocess import plot_cae_transformer_prediction
from src.dataset import create_transformer_dataset
from src.dataset import create_cae_dataset
from src.model_cae import CaeNet
from src import plot_train_loss
from src.model_transformer import Informer
from cae_prediction import cae_prediction


np.random.seed(0)
paddle.seed(0)


def load_yaml_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cae_transformer_prediction(encoded):
    # prepare params
    config = load_yaml_config(args.config_file_path)
    cae_data_params = config["cae_data"]
    transformer_data_params = config["transformer_data"]
    cae_model_params = config["cae_model"]
    transformer_model_params = config["transformer"]
    prediction_params = config["prediction"]

    # prepare network
    transformer = Informer(**transformer_model_params)
    transformer.set_state_dict(paddle.load(prediction_params["transformer_ckpt_path"]))

    cae = CaeNet(
        cae_model_params["data_dimension"],
        cae_model_params["conv_kernel_size"],
        cae_model_params["maxpool_kernel_size"],
        cae_model_params["maxpool_stride"],
        cae_model_params["encoder_channels"],
        cae_model_params["decoder_channels"],
        cae_model_params["channels_dense"],
    )
    cae.set_state_dict(paddle.load(prediction_params["cae_ckpt_path"]))

    # prepare dataset
    _, input_seq = create_transformer_dataset(
        encoded,
        transformer_data_params["batch_size"],
        transformer_data_params["time_size"],
        transformer_data_params["latent_size"],
        transformer_data_params["time_window"],
        transformer_data_params["gaussian_filter_sigma"],
    )

    _, true_data = create_cae_dataset(
        cae_data_params["data_path"], cae_data_params["batch_size"]
    )

    output_seq_pred = np.zeros(
        shape=(
            transformer_data_params["time_size"]
            - transformer_data_params["time_window"],
            transformer_data_params["latent_size"],
        )
    )

    print(f"=================Start transformer prediction=====================")
    input_enc_pred = input_seq[0].reshape(
        (
            1,
            transformer_data_params["time_window"],
            transformer_data_params["latent_size"],
        )
    )
    input_enc_pred = input_enc_pred.astype(np.float32)
    dec_inp = paddle.zeros(
        (input_enc_pred.shape[0], transformer_data_params["pred_len"], input_enc_pred.shape[-1]), dtype='float32'
    )
    input_dec_pred = paddle.concat(
        [paddle.to_tensor(input_enc_pred[:, - transformer_data_params["label_len"]:, :]), dec_inp], axis=1)
    for sample in range(
            0, transformer_data_params["time_size"] - transformer_data_params["time_window"]
    ):
        output_seq_pred[sample, :] = transformer(paddle.to_tensor(input_enc_pred), input_dec_pred).numpy()[
                                     0, 0, :
                                     ]
        input_enc_pred[0, :-1, :] = input_enc_pred[0, 1:, :]
        input_enc_pred[0, -1, :] = output_seq_pred[sample, :]
        dec_inp = paddle.zeros(
            (input_enc_pred.shape[0], transformer_data_params["pred_len"], input_enc_pred.shape[-1]), dtype='float32'
        )
        input_dec_pred = paddle.concat(
            [paddle.to_tensor(input_enc_pred[:, - transformer_data_params["label_len"]:, :]), dec_inp], axis=1)

    print(f"===================End lstm prediction====================")
    lstm_latent = np.expand_dims(output_seq_pred, 1)
    lstm_latent = paddle.to_tensor(lstm_latent.astype(np.float32))
    cae_lstm_predict_time = (
            transformer_data_params["time_size"] - transformer_data_params["time_window"]
    )
    cae_lstm_predict = np.zeros(
        (cae_lstm_predict_time, true_data.shape[1], true_data.shape[2])
    )
    for i in range(prediction_params["decoder_data_split"]):
        time_predict_start, time_predict_end = (
            prediction_params["decoder_time_spilt"][i],
            prediction_params["decoder_time_spilt"][i + 1],
        )
        cae_lstm_predict[time_predict_start:time_predict_end] = np.squeeze(
            cae.decoder(lstm_latent[time_predict_start:time_predict_end]).numpy()
        )
    plot_cae_transformer_prediction(
        lstm_latent,
        cae_lstm_predict,
        true_data,
        prediction_params["prediction_result_dir"],
        transformer_data_params["time_size"],
        transformer_data_params["time_window"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cae-lstm prediction")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    print(f"pid:{os.getpid()}")
    cae_latent = cae_prediction(args.config_file_path)
    cae_transformer_prediction(cae_latent)
