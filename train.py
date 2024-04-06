import argparse
import os
import sys
import shutil
import json
import glob
import signal
import pickle
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from data_loader import VideoLoader
from callbacks import PlotLearning, MonitorLRDecay, AverageMeter
from model import GestureDetection
from torchvision.transforms import *
import logging

str2bool = lambda x: (str(x).lower() == 'true')

parser = argparse.ArgumentParser(
    description='PyTorch Jester Training using JPEG')
parser.add_argument('--config', '-c', help='json config file path')
parser.add_argument('--eval_only', '-e', default=False, type=str2bool,
                    help="evaluate trained model on validation data.")
parser.add_argument('--resume', '-r', default=False, type=str2bool,
                    help="resume training from given checkpoint.")
parser.add_argument('--use_gpu', default=True, type=str2bool,
                    help="flag to use gpu or not.")
parser.add_argument('--gpus', '-g', help="gpu ids for use.")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

if args.use_gpu:
    gpus = [int(i) for i in args.gpus.split(',')]
    print("=> active GPUs: {}".format(args.gpus))

best_prec1 = 0

# load config file
with open(args.config) as data_file:
    config = json.load(data_file)


def main():
    try:
        global args, best_prec1
        model_name = config["model_name"]
        output_dir = config["output_dir"]
        print("=> Output folder for this run -- {}".format(model_name))
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, 'plots'))

        def signal_handler(signal, frame):
            try:
                num_files = len(glob.glob(save_dir + "/*"))
                if num_files < 1:
                    shutil.rmtree(save_dir)
                print('You pressed Ctrl+C!')
                sys.exit(0)
            except Exception as ex:
                logging.warning(ex)


        signal.signal(signal.SIGINT, signal_handler)
        model = GestureDetection(config['num_classes'])

        if args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

        if args.resume:
            if os.path.isfile(config['checkpoint']):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(config['checkpoint'])
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config['checkpoint'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(
                    config['checkpoint']))

        transform = Compose([
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])

        train_data = VideoLoader(root=config['train_data_folder'],
                                csv_file_input=config['train_data_csv'],
                                csv_file_labels=config['labels_csv'],
                                clip_size=config['clip_size'],
                                nclips=1,
                                step_size=config['step_size'],
                                is_val=False,
                                transform=transform,
                                )

        print(" > Using {} processes for data loader.".format(
            config["num_workers"]))
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=True)

        val_data = VideoLoader(root=config['val_data_folder'],
                            csv_file_input=config['val_data_csv'],
                            csv_file_labels=config['labels_csv'],
                            clip_size=config['clip_size'],
                            nclips=1,
                            step_size=config['step_size'],
                            is_val=True,
                            transform=transform,
                            )

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=False)

        assert len(train_data.classes) == config["num_classes"]

        criterion = nn.CrossEntropyLoss().to(device)

        lr = config["lr"]
        last_lr = config["last_lr"]
        momentum = config['momentum']
        weight_decay = config['weight_decay']

        optimizer = torch.optim.AdamW(model.parameters(), lr,
                                    weight_decay=weight_decay)

        

        if args.eval_only:
            validate(val_loader, model, criterion, train_data.classes_dict)
            return


        plotter = PlotLearning(os.path.join(
            save_dir, "plots"), config["num_classes"])
        lr_decayer = MonitorLRDecay(0.6, 3)
        val_loss = 9999999


        num_epochs = int(config["num_epochs"])
        if num_epochs == -1:
            num_epochs = 999999

        print(" > Training is getting started...")
        print(" > Training takes {} epochs.".format(num_epochs))
        start_epoch = args.start_epoch if args.resume else 0

        for epoch in range(start_epoch, num_epochs):
            lr = lr_decayer(val_loss, lr)
            print(" > Current LR : {}".format(lr))

            if lr < last_lr and last_lr > 0:
                print(" > Training is done by reaching the last learning rate {}".
                    format(last_lr))
                sys.exit(1)


            train_loss, train_top1, train_top5 = train(
                train_loader, model, criterion, optimizer, epoch)

            val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)

            plotter_dict = {}
            plotter_dict['loss'] = train_loss
            plotter_dict['val_loss'] = val_loss
            plotter_dict['acc'] = train_top1
            plotter_dict['val_acc'] = val_top1
            plotter_dict['learning_rate'] = lr
            plotter.plot(plotter_dict)
            is_best = val_top1 > best_prec1
            best_prec1 = max(val_top1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "Conv4Col",
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, config)
    except Exception as ex:
        logging.warning(ex)


def train(train_loader, model, criterion, optimizer, epoch):
    try:
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.train()

        for i, (input, target) in tqdm(enumerate(train_loader)):
            input, target = input.to(device), target.to(device)
            model.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.detach(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % config["print_freq"] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
        return losses.avg, top1.avg, top5.avg
    except Exception as ex:
        logging.warning(ex)


def validate(val_loader, model, criterion, class_to_idx=None):
    try:
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.eval()

        logits_matrix = []
        targets_list = []

        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss = criterion(output, target)
                if args.eval_only:
                    logits_matrix.append(output.detach().cpu().numpy())
                    targets_list.append(target.detach().cpu().numpy())
                prec1, prec5 = accuracy(output.detach(), target.detach().cpu(), topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))
                if i % config["print_freq"] == 0:
                    print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), loss=losses, top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

            if args.eval_only:
                logits_matrix = np.concatenate(logits_matrix)
                targets_list = np.concatenate(targets_list)
                print(logits_matrix.shape, targets_list.shape)
                save_results(logits_matrix, targets_list, class_to_idx, config)
            return losses.avg, top1.avg, top5.avg
    except Exception as ex:
        logging.warning(ex)

def save_results(logits_matrix, targets_list, class_to_idx, config):
    try:
        path_to_save = os.path.join(
            config['output_dir'], config['model_name'], "test_results.pkl")
        with open(path_to_save, "wb") as f:
            pickle.dump([logits_matrix, targets_list, class_to_idx], f)
    except Exception as ex:
        logging.warning(ex)


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    try:
        checkpoint_path = os.path.join(
            config['output_dir'], config['model_name'], filename)
        model_path = os.path.join(
            config['output_dir'], config['model_name'], 'model_best.pth.tar')
        torch.save(state, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, model_path)
    except Exception as ex:
        logging.warning(ex)


def accuracy(output, target, topk=(1,)):
    try:
        maxk = max(topk)
        batch_size,res = target.size(0),[]

        _, pred = output.cpu().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    except Exception as ex:
        logging.warning(ex)


if __name__ == '__main__':
    main()


