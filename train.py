from __future__ import print_function
import argparse
from math import log10

from libs.GANet.modules.GANet import MyLoss2
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from models.GANet_deep import GANet
import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set
import numpy as np

from robust_loss_pytorch.decoupled_adaptive import DecoupledAdaptiveLossFunction
from utils.utils import image_var
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right,
                             opt.kitti, opt.kitti2015, opt.shift)
test_set = get_test_set(opt.data_path, opt.val_list, [576, 960], opt.left_right, opt.kitti, opt.kitti2015)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)
adaptive_loss = DecoupledAdaptiveLossFunction(num_dims=16,
                                              float_dtype=np.float32,
                                              device='cuda:0')

# criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()

model_parameters = list(model.parameters())
loss_parameters = list(adaptive_loss.parameters())
optimizer = optim.Adam(model_parameters + loss_parameters, lr=opt.lr, betas=(0.9, 0.999))
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        adaptive_loss.load_state_dict(checkpoint['loss'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

train_writer = SummaryWriter(opt.save_path)


def train(epoch):
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    valid_iteration = 0
    model.train()
    adaptive_loss.train()

    num_iter = 0
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1],
                                                                                  requires_grad=True), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        input1_label = image_var(input1, 9)

        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            optimizer.zero_grad()

            if opt.model == 'GANet11':
                disp1, disp2 = model(input1, input2)
                disp0 = (disp1 + disp2) / 2.
                # if opt.kitti or opt.kitti2015:
                #     loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(
                #         disp2[mask], target[mask])
                # else:
                #     loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(
                #         disp2[mask], target[mask], reduction='mean')

                bs = disp0.shape[0]
                # residual0 = disp0 - target
                residual1 = disp1 - target
                residual2 = disp2 - target
                # loss0 = adaptive_loss(residual0.view(bs, -1), input1_label.view(bs, -1)).view(residual0.shape)
                loss1 = adaptive_loss.lossfun(residual1.view(bs, -1), input1_label.view(bs, -1)).view(residual1.shape)
                loss2 = adaptive_loss.lossfun(residual2.view(bs, -1), input1_label.view(bs, -1)).view(residual2.shape)

                loss = 0.4 * loss1[mask].mean() + 1.2 * loss2[mask].mean()


            elif opt.model == 'GANet_deep':
                disp0, disp1, disp2 = model(input1, input2)
                # if opt.kitti or opt.kitti2015:
                #     loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                #         disp1[mask], target[mask], reduction='mean') + criterion(disp2[mask], target[mask])
                # else:
                #     loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                #         disp1[mask], target[mask], reduction='mean') + F.smooth_l1_loss(disp2[mask], target[mask],
                #                                                                         reduction='mean')

                bs = disp0.shape[0]
                residual0 = disp0 - target
                residual1 = disp1 - target
                residual2 = disp2 - target
                loss0 = adaptive_loss.lossfun(residual0.view(bs, -1), input1_label.view(bs, -1)).view(residual0.shape)
                loss1 = adaptive_loss.lossfun(residual1.view(bs, -1), input1_label.view(bs, -1)).view(residual1.shape)
                loss2 = adaptive_loss.lossfun(residual2.view(bs, -1), input1_label.view(bs, -1)).view(residual2.shape)

                loss = 0.4 * loss0[mask].mean() + 0.6 * loss1[mask].mean() + 1.2 * loss2[mask].mean()
            else:
                raise Exception("No suitable model found ...")

            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask]))
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f})".format(epoch, iteration,
                                                                                              len(training_data_loader),
                                                                                              loss.item(),
                                                                                              error0.item(),
                                                                                              error1.item(),
                                                                                              error2.item()))
            sys.stdout.flush()

            train_writer.add_scalar('train/error0', error0.item(), num_iter)
            train_writer.add_scalar('train/error1', error1.item(), num_iter)
            train_writer.add_scalar('train/error2', error2.item(), num_iter)
            train_writer.add_scalar('train/loss', loss.item(), num_iter)
            train_writer.add_scalars('train/alpha',
                                     {f"alpha_{i}": adaptive_loss.latent_alpha.detach().cpu().numpy()[0, i] for i in
                                      range(16)}, num_iter)
            train_writer.add_scalars('train/scale',
                                     {f"scale_{i}": adaptive_loss.latent_scale.detach().cpu().numpy()[0, i] for i in
                                      range(16)}, num_iter)

            num_iter += 1

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch,
                                                                                                 epoch_loss / valid_iteration,
                                                                                                 epoch_error0 / valid_iteration,
                                                                                                 epoch_error1 / valid_iteration,
                                                                                                 epoch_error2 / valid_iteration))


def val():
    epoch_error2 = 0

    valid_iteration = 0
    model.eval()
    adaptive_loss.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=False), Variable(batch[1],
                                                                                   requires_grad=False), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            with torch.no_grad():
                disp2 = model(input1, input2)
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                valid_iteration += 1
                epoch_error2 += error2.item()
                print("===> Test({}/{}): Error: ({:.4f})".format(iteration, len(testing_data_loader), error2.item()))



    print("===> Test: Avg. Error: ({:.4f})".format(epoch_error2 / valid_iteration))

    return epoch_error2 / valid_iteration


def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best.pth')
    print("Checkpoint saved to {}".format(filename))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
        lr = opt.lr
    else:
        lr = opt.lr * 0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    error = 10000
    for epoch in range(1, opt.nEpochs + 1):
        #        if opt.kitti or opt.kitti2015:
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        is_best = False

        val_error=val()
        train_writer.add_scalar('val/epe', val_error, epoch)
        if val_error < error:
           error=val_error
           is_best = True

        if opt.kitti or opt.kitti2015:
            if epoch % 50 == 0 and epoch >= 300:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': adaptive_loss.state_dict(),
                }, is_best)
        else:
            if epoch >= 8:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': adaptive_loss.state_dict(),
                }, is_best)

    save_checkpoint(opt.save_path, opt.nEpochs, {
        'epoch': opt.nEpochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': adaptive_loss.state_dict(),
    }, is_best)
