import torch.nn as nn
import torch
class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()
        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential( 
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            #nn.Conv2d(512, 512, 1),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64), 
            *upsample(64, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

def init_weight(network):
    classname = network.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(network.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(network.weight.data, 1.0, 0.02)
        nn.init.constant_(network.bias.data, 0.0)

def save_model(model, model_name, model_dir, optimizer, index):
    state = {'net': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': index}
    torch.save(state, model_dir + model_name + str(index) + '.pth')

def load_model(model, model_dir):
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['net'])

if __name__ == "__main__":
    """for test"""
    im = torch.randn(4, 3, 128, 128)
    model = Discriminator()

    print(list(model.children()))
    import time
    t = time.time()
    x = model(im)
    print(time.time() - t)
    print(x.shape)
    del model
    del x