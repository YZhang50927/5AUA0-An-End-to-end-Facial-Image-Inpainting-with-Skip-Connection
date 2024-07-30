import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

from evaluation.lpips import LPIPS
from models import *
from config import *
from data_loader import test_loader

cfg = get_cfg()
plt.switch_backend('agg')

def evalution(net_type: str = 'alex', version: str = '0.1'):
    device = torch.device("cuda" if cfg.use_cuda else "cpu")
    lpips_loss = []
    generator = Generator().to(device)
    criterion = LPIPS(net_type, version).to(device)
    for epoch in range(cfg.epoches + 1):
        load_model(generator, cfg.model_path+'generator'+str(epoch)+'.pth')
        loss = 0
        cnt = 0
        for imgs, masked_imgs, loc in tqdm(test_loader):
            imgs = Variable(imgs.to(device))
            masked_imgs = Variable(masked_imgs.to(device))
            loc = loc[0].item()  # Upper-left location of mask
            gen_mask = generator(masked_imgs)
            original_part = imgs[:, :, loc : loc + cfg.mask_size, loc : loc + cfg.mask_size]
            gen_part = gen_mask[:, :, loc : loc + cfg.mask_size, loc : loc + cfg.mask_size]
            loss = loss + criterion(gen_mask, original_part).item()
            cnt = cnt + 1           
        lpips_loss.append(loss/cnt)
        print("[Model: %d][loss %f]"%(epoch, loss/cnt))
    
    np.savetxt(cfg.save_file + 'LPIPS_loss.txt', np.array(lpips_loss))

def plot_save(variable_name, data_dir):
    data = np.loadtxt(data_dir+variable_name+'.txt')
    plt.plot(data)
    plt.xlabel('epoch')
    plt.ylabel('performance')
    plt.legend([variable_name])
    x_major_locator = MultipleLocator(1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)   
    plt.xlim(-0.5,cfg.epoches)
    plt.savefig(data_dir+'performance.jpg')
  
if __name__ == "__main__":
    evalution()
    plot_save("LPIPS_loss", cfg.save_file)
