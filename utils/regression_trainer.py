from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
import scipy
import scipy.stats
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.rape import Rape
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob

from models.RapeNet import net
# from models.RapeNet_plus import net



def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rmse = 0
    return rmae, rmse


def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value**2

def plot_learning_curves(epoch_loss, epoch_mse, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epoch_loss, label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epoch_mse, label='train mse', color='tab:orange')
    ax2.legend(loc = 'upper right')
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()

class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Rape(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        self.model = net()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_rmae = np.inf
        self.best_rmse = np.inf
        self.best_r2 = np.inf
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rmae = AverageMeter()
        epoch_rmse = AverageMeter()
        epoch_r2 = AverageMeter()
        train_loss = []
        train_mae = []
        train_mse = []

        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                rmae, rmse = compute_relerr(pre_count, gd_count)
                r2 = rsquared(pre_count, gd_count)

                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
                epoch_rmae.update(rmae, N)
                epoch_rmse.update(rmse, N)
                epoch_r2.update(r2, N)
                train_loss.append(epoch_loss.get_avg())
                train_mae.append(epoch_mae.get_avg())
                train_mse.append(epoch_mse.get_avg())

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()),
                             epoch_mae.get_avg(),time.time()-epoch_start))

        plot_learning_curves(train_loss, train_mse, self.save_dir)
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        epoch_gt = []
        epoch_pd = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                epoch_gt.append(count[0].item())
                epoch_pd.append(torch.sum(outputs).item())
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        rmae, rmse = compute_relerr(epoch_pd, epoch_gt)
        r2 = rsquared(epoch_pd, epoch_gt)


        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f} RMSE: {:.2f} RMAE: {:.2f} R2: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, rmse, rmae, r2, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.best_rmse = rmse
            self.best_rmae = rmae
            self.best_r2 = r2
            logging.info("save best mse {:.2f} mae {:.2f} rmse {:.2f} rmae {:.2f} r2 {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae, self.best_rmse, self.best_rmae,
                                                                                 self.best_r2, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))