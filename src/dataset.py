import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

class CreateDataset(Dataset):
    """convert raw data into train dataset"""

    def __init__(self, data, label):
        super(CreateDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def create_cae_dataset(data_path, batch_size):
    """create cae dataset"""
    true_data = np.load(data_path)
    train_data = np.expand_dims(true_data, 1).astype(np.float32)

    dataset = CreateDataset(train_data, train_data)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return loader, true_data

def create_transformer_dataset(latent_true, batch_size, time_size, latent_size, time_window, gaussian_filter_sigma):
    """create transformer dataset"""
    latent_true = np.squeeze(latent_true)
    latent_true = latent_true.astype(np.float32)
    encoded_f = np.copy(latent_true).astype(np.float32)

    for i in range(latent_size):
        encoded_f[:, i] = gaussian_filter1d(encoded_f[:, i], sigma=gaussian_filter_sigma)

    print("Shape of encoded_f after Gaussian filter:", encoded_f.shape)  # 添加的打印语句

    input_seq = np.zeros(shape=(time_size - time_window, time_window, latent_size)).astype(np.float32)
    output_seq = np.zeros(shape=(time_size - time_window, 1, latent_size)).astype(np.float32)

    print("Shape of input_seq and output_seq:", input_seq.shape, output_seq.shape)  # 添加的打印语句

    sample = 0
    for t in range(time_window, time_size):
        input_seq[sample, :, :] = encoded_f[t - time_window : t, :]
        output_seq[sample, 0, :] = encoded_f[t, :]
        sample = sample + 1

        if sample % 100 == 0:  # 每100个样本打印一次
            print(f"Sample {sample}, t {t}, Shape of input_seq[sample] and output_seq[sample]:",
                  input_seq[sample].shape, output_seq[sample].shape)  # 添加的打印语句

    dataset = CreateDataset(input_seq, output_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    print(f"Shape of output: {output_seq.shape,input_seq.shape}")

    return dataloader, input_seq



