import data
import torch
import model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import SimpleITK as sitk
import torch.nn.functional as F
import argparse
from tqdm import tqdm


if __name__ == '__main__':

    writer = SummaryWriter('./log')

    #--------------------------
    #   Setting parameters
    #--------------------------

    parser = argparse.ArgumentParser()

    # path of training and test data
    parser.add_argument('-train_path',type = str, default='./data/train.nii.gz',
                        help = 'the file path of input train noisy data')
    parser.add_argument('-val_path', type = str, default= './data/val.nii.gz',
                        help = 'the file path of input validation noisy data')
    parser.add_argument('-val_gt_path', type = str, default='./data/val_gt.nii.gz',
                        help = 'the file path of input validation ground truth data')
    parser.add_argument('-model_save_path',type = str, default='model/',
                        help = 'the path of saved model weight')
    parser.add_argument('-save_path',type = str, default='output/',
                        help = 'the path of validation output')


    # training parameters
    parser.add_argument('-lr', type = float, default=1e-4, dest= 'lr',
                        help = 'learning rate')
    parser.add_argument('-epoch', type = int, default= 2000, dest= 'epoch',
                        help = 'the total number of training epoches')
    parser.add_argument('-summary_epoch', type = int, default=250, dest='summary_epoch',
                        help = 'the current model will be saved per summary_epoch')
    parser.add_argument('-bs', type = int, default = 16, dest='batch_size',
                        help = 'the number of training batches')
    parser.add_argument('-ps', type = int, default=128, dest= 'patch_size',
                        help = 'the patch size for training images')
    parser.add_argument('-gpu',type = int, default=0,dest='gpu',
                        help = 'the number of GPU')

    args = parser.parse_args()
    model_save_path = args.model_save_path
    save_path = args.save_path
    lr = args.lr
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    patch_size = args.patch_size

    utils.makedir(model_save_path)
    utils.makedir(save_path)


    DEVICE = torch.device('cuda:{}'.format(str(args.gpu) if torch.cuda.is_available() else 'cpu'))

    # load training and validation data
    train_data = sitk.GetArrayFromImage(sitk.ReadImage(args.train_path))
    val_data = sitk.GetArrayFromImage(sitk.ReadImage(args.val_path))
    val_gt = sitk.GetArrayFromImage(sitk.ReadImage(args.val_gt_path))

    val_data = val_data/255.
    val_gt = val_gt/255.

    print(train_data.shape)
    print(val_data.shape)

    val_H = val_data.shape[1]
    val_W = val_data.shape[2]

    Sub_sampler = utils.Sub_sampler(2, DEVICE)

    train_loader = data.loader_train(train_data, patch_size=patch_size, batch_size=batch_size)
    val_loader = data.loader_val(val_data, batch_size=10)

    net = model.Noise2SR(in_c=1, out_c=1, scale_factor=2, feature_dim=64).to(DEVICE)

    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr = lr)

    tqdm_loop = tqdm(range(epoch), ncols = 120)

    for e in tqdm_loop:

        net.train()
        loss_train = 0

        # train mode
        for i, (x1) in enumerate(train_loader):

            N, H, W = x1.shape
            x1 = x1.to(DEVICE).unsqueeze(1)
            x1_in, mask = Sub_sampler.sample_img(x1, 1)
            mask = mask.to(DEVICE)
            img_pred = net(x1_in.float())

            img_pred_mask = img_pred * mask
            x1_mask = x1 * mask

            loss = loss_fun(img_pred_mask.float(), x1_mask.float())
            loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record and print loss
            loss_train += loss.item()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tqdm_loop.set_description_str('(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.6f}'.format(e + 1,
                                                                                      epoch,
                                                                                      i + 1,
                                                                                      len(train_loader),
                                                                                      current_lr,
                                                                                      loss.item()))

        writer.add_scalar('MSE_train', loss_train / len(train_loader), e + 1)

        if (e + 1) % summary_epoch == 0:

            net.eval()

            with torch.no_grad():

                loss_val = 0
                pred_whole = []

                # evaluation mode use the whole noisy image to get clean-image
                for i, (x1) in enumerate(val_loader):
                    N, H, W = x1.shape
                    x1_in = x1.to(DEVICE).float().unsqueeze(1)
                    pred = net(x1_in)
                    img_pred = F.interpolate(pred,
                                             scale_factor=0.5,
                                             mode = 'bilinear',
                                             align_corners=True)
                    img_pred = img_pred.squeeze(1)
                    pred_whole.append(np.array(img_pred.cpu()))

                pred_whole = np.concatenate(pred_whole, axis=0).reshape((-1, val_H, val_W))

                psnr_value = utils.PSNR(pred_whole, val_gt)
                ssim_value = utils.SSIM(pred_whole, val_gt)
                print('PSNR:{:.2f}'.format(np.mean(psnr_value)))
                print('PSNR:{:.4f}'.format(np.mean(ssim_value)))

                print(pred_whole.shape)

                pred_whole_img = sitk.GetImageFromArray(pred_whole)
                sitk.WriteImage(pred_whole_img, save_path + 'N2SR_denoised_{}.nii.gz'.format(e+1))
                torch.save(net.state_dict(), model_save_path + 'model_param_{}.pkl'.format(e + 1))


    writer.flush()















