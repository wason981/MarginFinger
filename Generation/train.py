import argparse
import os
import copy
import pprint
from os import path
from torchvision.utils import save_image
import torch
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import  get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models, build_optimizers)
os.environ['CUDA_VISIBLE_DEVICES'] ='2'

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=150, help='number of epochs to run before terminating')
parser.add_argument('--model_epoch', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')
parser.add_argument('--d', type=float, default=0.1,help='distance  to the classification boundary')
parser.add_argument('--omega', type=float, default=5, help='condition loss weight')
args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
outdir = config['training']['out_dir'] + f'/d{args.d}omega{args.omega}'
out_dir = config['training']['out_dir']+ f'/d{args.d}omega{args.omega}' if args.outdir is None else args.outdir + f'/d{args.d}omega{args.omega}'


def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator': config['discriminator'],
        'training': config['training']
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    log_every = config['training']['log_every']
    sample_every = config['training']['sample_every']
    backup_every = config['training']['backup_every']

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    device = torch.device("cuda:0" if is_cuda else "cpu")

    train_dataset, _ = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    # Create models
    generator, discriminator = build_models(config)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for name, module in discriminator.named_modules():
        if isinstance(module, torch.nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

    devices = [int(x) for x in args.devices]
    generator = torch.nn.DataParallel(generator, device_ids=devices)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator,
                                   discriminator=discriminator,
                                   )
    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    x_test, y_test = x_test.to(device), y_test.to(device)
    z_test = zdist.sample((ntest, ))
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)

    # Test generator
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]
        for model in [generator, discriminator]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print('Batch norm in discriminator not compatible with exponential moving average')
                        exit()
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    # Load checkpoint if it exists
    epoch_idx = utils.get_most_recent(checkpoint_dir, 'model') if args.model_epoch == -1 else args.model_epoch
    checkpoint_io.load_models(epoch_idx=epoch_idx)

    # Evaluator
    evaluator = Evaluator(
        generator_test,
        zdist,
        train_loader=train_loader,
        batch_size=batch_size,
        device=device,
        )

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      g_optimizer,
                      d_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'],
                      c=args.d,
                      omega=args.omega,
                      dataset=config['data']['type'],
                      )

    # Training loop
    print('Start training...')
    while epoch_idx < args.nepochs:
        epoch_idx += 1
        print('Epoch {}'.format(epoch_idx))
        for x_real, y in train_loader:
            x_real, y = x_real.to(device), y.to(device)
            z = zdist.sample((batch_size, ))

            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)

            # Generators updates
            gloss = trainer.generator_trainstep(y, z)

            if config['training']['take_model_average']:
                update_average(generator_test, generator, beta=config['training']['model_average_beta'])

        #(i) Sample if necessary
        if epoch_idx % sample_every == 0 and epoch_idx > 0:
            x = evaluator.create_samples(z_test, y_test)
            logger.add_imgs(x, 'all', epoch_idx)

        # (ii)Print stats
        if epoch_idx % log_every == 0 :
            logger.add('losses', 'discriminator', dloss, epoch_idx=epoch_idx)
            logger.add('losses', 'regularizer', reg, epoch_idx=epoch_idx)
            logger.add('losses', 'generator', gloss, epoch_idx=epoch_idx)
            g_loss_last = logger.get_last('losses', 'generator')
            d_loss_last = logger.get_last('losses', 'discriminator')
            d_reg_last = logger.get_last('losses', 'regularizer')
            print('[epoch %0d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                  % (epoch_idx, g_loss_last, d_loss_last, d_reg_last))

        # (iii) Backup if necessary
        if epoch_idx % backup_every == 0 :
            # print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % epoch_idx)
            logger.save_stats('stats_%08d.p' % epoch_idx)

        if epoch_idx >= 0:
            checkpoint_io.save('model.pt', it=epoch_idx)

if __name__ == '__main__':
    main()
