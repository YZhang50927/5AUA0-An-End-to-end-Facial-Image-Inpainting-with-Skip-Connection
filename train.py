import os
from torchvision.utils import save_image
from tqdm import tqdm
from torch.autograd import Variable
import torch

from datasets import *
from models import *
from config import *
from data_loader import train_loader, test_loader

def save_output(device, epoch, generator):
    """Save the generated imgs, original imgs and masked imgs"""
    imgs, masked_imgs, loc = next(iter(test_loader))
    imgs = Variable(imgs.to(device))
    masked_imgs = Variable(masked_imgs.to(device))
    loc = loc[0].item()  # Upper-left location of mask
    gen_mask = generator(masked_imgs)
    #outputs = torch.cat((masked_imgs.data, gen_mask.data, imgs.data), -2)
    #save_image(outputs, "images/%d.png" % epoch, nrow=6, normalize=True)
    filled_imgs = masked_imgs.clone()
    filled_imgs[:, :, loc : loc + cfg.mask_size, loc : loc + cfg.mask_size] = gen_mask
    outputs = torch.cat((masked_imgs.data, filled_imgs.data, imgs.data), -2)	    
    save_image(outputs, "images/%d.png" % epoch, nrow=6, normalize=True)
  

def train(generator, discriminator, train_loader, adversarial_loss, reconstr_loss, optimizer_G, optimizer_D, patch, tensor):
    for imgs, masked_imgs, masked_parts in tqdm(train_loader):
        # Adversarial ground truths
        valid = Variable(tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(tensor))
        masked_imgs = Variable(masked_imgs.type(tensor))
        masked_parts = Variable(masked_parts.type(tensor))

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        gen_parts = generator(masked_imgs) # Generate a batch of images

        # Adversarial and reconstruction loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_reconstr = reconstr_loss(gen_parts, masked_parts)
        g_loss = 0.001 * g_adv + 0.999 * g_reconstr # Total loss

        g_loss.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability of classification
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    return d_loss, g_adv, g_reconstr

def main():

    os.makedirs("images", exist_ok=True)
    cfg = get_cfg()
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    tensor = torch.cuda.FloatTensor if cfg.use_cuda else torch.FloatTensor

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(cfg.mask_size / 2 ** 3), int(cfg.mask_size / 2 ** 3)
    patch = (1, patch_h, patch_w)

    # Loss function
    adversarial_loss = torch.nn.MSELoss().to(device)
    reconstr_loss = torch.nn.L1Loss().to(device)

    # Initialize generator and discriminator
    generator = Generator(channels=cfg.channels).to(device)
    discriminator = Discriminator(channels=cfg.channels).to(device)

    # Initialize weights
    generator.apply(init_weight)
    discriminator.apply(init_weight)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    
    #initial img
    save_output(device, 0, generator)
    save_model(generator, "generator", cfg.model_path, optimizer_G, 0)
    for epoch in range(cfg.epoches):
        d_loss, g_adv, g_reconstr = train(generator, discriminator, train_loader, adversarial_loss, reconstr_loss, optimizer_G, optimizer_D, patch, tensor)

        print(
            "[Epoch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, cfg.epoches, d_loss.item(), g_adv.item(), g_reconstr.item())
        )

        save_output(device, epoch + 1, generator) #save output imgs for every epoch
        save_model(generator, "generator", cfg.model_path, optimizer_G, epoch + 1)
    
    #save the last tranined model
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path, exist_ok=True)

if __name__ == '__main__':
    main()
