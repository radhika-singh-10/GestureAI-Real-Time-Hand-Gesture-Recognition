import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from torch.optim.optimizer import Optimizer
import logging



class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer is currently not an instance of torch.optim.Optimizer")
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau class does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        self._reset()

    def _reset(self):
        try:
            if self.mode not in ['min', 'max']:
                raise ValueError('Learning Rate Plateau Reducing mode %s is unknown!' % self.mode)
            if self.mode == 'min':
                self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
                self.best = np.Inf
            else:
                self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
                self.best = -np.Inf
            self.cooldown_counter = 0
            self.wait = 0
            self.lr_epsilon = self.min_lr * 1e-4
        except Exception as ex:
            logging.warning(f' _reset method exception {ex}')

    def reset(self):
        ##creating a resent model
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            raise ValueError('Learning Rate Plateau Reducing requires metrics available!')
        else:
            if self.in_cooldown():
                self.cooldown_counter = self.cooldown_counter - 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                logging.info('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait = self.wait + 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


class MonitorLRDecay:
    def __init__(self, decay, patience):
        if decay >= 1.0:
            raise ValueError('Decay factor should be less than 1.0.')
        self.best_loss = 999999
        self.decay = decay
        self.patience = patience
        self.count = 0
    #early stopping logic on the validation loss 
    def __call__(self, current_loss, current_lr):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.count = 0

        elif self.count > self.patience:
            current_lr = current_loss  * self.decay
            logging.info(f" > New learning rate -- {current_lr}")
            self.count = 0
        else:
            self.count = self.count + 1
        return current_lr


class PlotLearning:
    def __init__(self, save_path, num_classes):
        if not os.path.exists(save_path):
            raise FileNotFoundError("Save path does not exist.")
        self.init_loss = 5
        self.accuracy, self.val_accuracy, self.losses, self.val_losses,self.learning_rates = [], [], [], [], []
        self.save_path_loss = os.path.join(save_path, 'loss_plot.png')
        self.save_path_accu = os.path.join(save_path, 'accu_plot.png')
        self.save_path_lr = os.path.join(save_path, 'lr_plot.png')
        

    def plot(self, logs):
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        best_val_acc = max(self.val_accuracy)
        best_train_acc = max(self.accuracy)
        best_val_epoch = self.val_accuracy.index(best_val_acc)
        best_train_epoch = self.accuracy.index(best_train_acc)

        plt.figure(1)
        plt.gca().cla()
        plt.plot(self.accuracy, label='train')
        plt.plot(self.val_accuracy, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accu)

        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.plot(self.losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.savefig(self.save_path_loss)

        self.learning_rates.append(logs.get('learning_rate'))

        min_learning_rate = min(self.learning_rates)
        max_learning_rate = max(self.learning_rates)

        plt.figure(2)
        plt.gca().cla()
        plt.plot(self.learning_rates)
        plt.title("max_learning_rate-{0:.6f}, min_learning_rate-{1:.6f}".format(max_learning_rate, min_learning_rate))
        plt.savefig(self.save_path_lr)


class ProgressTracker:
    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        values = values or []
        self._update_sum_values(values, current)
        self.seen_so_far

    def _update_sum_values(self, values, current):
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  
    optimizer = Optimizer()  
    lr_scheduler = ReduceLROnPlateau(optimizer)
    for epoch in range(1, 20):
        metrics = np.random.rand()
        lr_scheduler.step(metrics, epoch)

    monitor = MonitorLRDecay(decay=0.5, patience=5)
    current_lr = 0.01 
    for epoch in range(1, 20):
        loss = np.random.rand()
        current_lr = monitor(loss, current_lr)

    plotter = PlotLearning(save_path='plots', num_classes=10)
    logs = {'acc': 0.85, 'val_acc': 0.82, 'loss': 0.4, 'val_loss': 0.5, 'learning_rate': 0.001}
    plotter.plot(logs)

    progress_tracker = ProgressTracker(target=100, width=30)
    for i in range(1, 101):
        progress_tracker.update(i)

    meter = AverageMeter()
    for i in range(1, 11):
        meter.update(i)
    print("Average Meter :", meter.avg)
