import paddle
import numpy as np

def trans_shape(trans_index, tensor):
    tensor = paddle.transpose(tensor, perm=trans_index)
    return tensor

def mask_fill(mask, data, num):
    data = paddle.where(mask, paddle.full_like(data, num), data)
    return data

def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {}   # set a default value
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)
        print('Updating learning rate to {}'.format(lr))
    return optimizer

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        paddle.save(model.state_dict(), path+'/'+'checkpoint.ckpt')
        self.val_loss_min = val_loss

class dotdict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = paddle.mean(data, axis=0)
        self.std = paddle.std(data, axis=0)

    def transform(self, data):
        if len(data.shape) > 1:  # 如果数据是二维的
            mean_tensor = paddle.reshape(self.mean, (1, -1))
            std_tensor = paddle.reshape(self.std, (1, -1))
        else:  # 如果数据是一维的
            mean_tensor = self.mean
            std_tensor = self.std
        return (data - mean_tensor) / std_tensor

    def inverse_transform(self, data):
        mean = paddle.to_tensor(self.mean, dtype=data.dtype) if isinstance(data, paddle.Tensor) else self.mean
        std = paddle.to_tensor(self.std, dtype=data.dtype) if isinstance(data, paddle.Tensor) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

