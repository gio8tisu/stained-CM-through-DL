import argparse
from cyclegan import cyclegan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--dataset-name', type=str, default='conf_data6', help='name of the dataset')
    parser.add_argument('--batch-size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--decay-epoch', type=int, default=100, help='epoch from which to start lr decay')
    parser.add_argument('--n-cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img-height', type=int, default=256, help='size of image height')
    parser.add_argument('--img-width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample-interval', type=int, default=100,
                        help='interval between sampling images from generators')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='interval between saving model checkpoints')
    parser.add_argument('--n-residual-blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--unet', action='store_true', help='unet generator')
    opt = parser.parse_args()
    print(opt)
    if opt.unet:
        cyclegan.main_unet(opt)
    else:
        cyclegan.main(opt)
