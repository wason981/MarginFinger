import sklearn
from gan_training.config import build_models,load_config#,build_optimizers
from torch import nn
from gan_training.checkpoints import CheckpointIO
from os import path
import torch
import copy
from gan_training.distributions import  get_zdist
import argparse
import os
from torchvision.utils import save_image
import numpy as np
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('--config', default='configs/margin/conditional.yaml',type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')

ckptdir=[
        'm1c0.3alpha1.0beta0.0gamma1.0omega4.0',
]

# ckptdir =args.ckptdir
is_cuda = True
device = torch.device("cuda:0" if is_cuda else "cpu")


generator, _ = build_models(config)
# g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)
generator = generator.to(device)

# generator = nn.DataParallel(generator, device_ids=devices)

generator.eval()
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
batch_size=1000
# zdist = torch.randn((batch_size, 128), device=device)
# zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
# zdist = torch.randn((batch_size, 128), device=device)
# zdist = zdist.sample((batch_size,))
# zdist = torch.randn((batch_size, 128), device=device)
#
# for i in range(10):
# z = zdist.sample((batch_size,))
# y= np.repeat(np.arange(0, 5), 200)
# y=torch.tensor(y,device=device,dtype=torch.int64)
# torch.save(z,'zm5.pt')
# torch.save(y,'ym5.pt')
z=torch.load('zm.pt')
y=torch.load('ym.pt')
for ckpt in ckptdir:
    print(ckpt)
    checkpoint_dir = path.join('output','margin','conditional',ckpt, 'chkpts')
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(generator=generator)
    # for it in range(5000, (len(os.listdir(checkpoint_dir))-3)*1000, 1000):
#     for pt in os.listdir(checkpoint_dir):
#         checkpoint_io.load_models(it=it)
#         os.makedirs(os.path.join('output','margin','conditional',ckpt,'imgs',f'{it}','samples'), exist_ok=True)
# logger = Logger(log_dir=path.join(out_dir, 'logs'),
#                 img_dir=path.join(out_dir, 'imgs'),
#                 monitoring=config['training']['monitoring'],
#                 monitoring_dir=path.join(out_dir, 'monitoring'))

    with torch.no_grad():
        for batch in range(10):
            x=generator(z[batch*100:(batch+1)*100], y[batch*100:(batch+1)*100])
            # x = generator(z, y)
            x = x / 2 + 0.5
            for j in range(x.shape[0]):
                save_image(copy.deepcopy(x[j]),os.path.join('output','margin','conditional',ckpt,'imgs',f'{it}','samples', f'%03d.png'%(batch*100+j)))


