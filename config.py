import argparse
import torch

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", type=int, default=40, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size in training")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--dataset_path", type=str, default="./data/img_align_celeba/img_align_celeba/")
    parser.add_argument("--model_path", type=str, default="./models/")
    parser.add_argument("--val_num", type=int, default=1000, help="the number of validation imgs")
    parser.add_argument("--test_bs", type=int, default=12, help="batch size in test")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--save_file', type=str, default="./logs/")
    cfg = parser.parse_args()
    cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()
    #print(cfg)

    return cfg

