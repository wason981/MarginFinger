import argparse
import os
import copy
import pprint
from os import path
from torchvision.utils import save_image
import torch
from torch import nn
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models, build_optimizers)

torch.backends.cudnn.benchmark = True

def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator': config['discriminator'],
        # 'clusterer': config['clusterer'],
        'training': config['training']
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    log_every = config['training']['log_every']
    sample_every = config['training']['sample_every']
    inception_every = config['training']['inception_every']
    backup_every = config['training']['backup_every']
    # sample_nlabels = config['training']['sample_nlabels']
    # nlabels = config['data']['nlabels']
    # sample_nlabels = min(nlabels, sample_nlabels)

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    num_label=config['generator']['nlabels']
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    z_sample = torch.load(f'zm{num_label}.pt')
    y_sample = torch.load(f'ym{num_label}.pt')

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
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

    devices = [int(x) for x in args.devices]
    generator = nn.DataParallel(generator, device_ids=devices)
    discriminator = nn.DataParallel(discriminator, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator,
                                   discriminator=discriminator,
                                   # g_optimizer=g_optimizer,
                                   # d_optimizer=d_optimizer
                                   )
    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    # ydist = get_ydist(nlabels, device=device)
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    # x_cluster, y_cluster = utils.get_nsamples(train_loader, config['clusterer']['nimgs'])
    x_test, y_test = x_test.to(device), y_test.to(device)
    z_test = zdist.sample((ntest, ))
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)

    # Test generator
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
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

    # clusterer = get_clusterer(config)(discriminator=discriminator,
    #                                   x_cluster=x_cluster,
    #                                   x_labels=y_cluster,
    #                                   gt_nlabels=config['data']['nlabels'],
    #                                   **config['clusterer']['kwargs'])

    # Load checkpoint if it exists
    epoch_idx = utils.get_most_recent(checkpoint_dir, 'model') if args.model_epoch == -1 else args.model_epoch
    checkpoint_io.load_models(epoch_idx=epoch_idx)

    # if loaded_clusterer is None:
        # print('Initializing new clusterer. The first clustering can be quite slow.')
        # clusterer.recluster(discriminator=discriminator)
        # checkpoint_io.save_clusterer(clusterer, it=0)
        # np.savez(os.path.join(checkpoint_dir, 'cluster_samples.npz'), x=x_cluster)
    # else:
    #     print('Using loaded clusterer')
    #     clusterer = loaded_clusterer
    # g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    # d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

    # Evaluator
    evaluator = Evaluator(
        generator_test,
        zdist,
        # ydist,
        train_loader=train_loader,
        # clusterer=clusterer,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'])
    # Learning rate anneling
    # g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=epoch_idx)
    # d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=epoch_idx)

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      g_optimizer,
                      d_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'],
                      c=args.d,
                      alpha=args.alpha,
                      beta=args.beta,
                      gamma=args.gamma,
                      omega=args.omega,
                      dataset=config['data']['type'],
                      )

    # Training loop
    print('Start training...')
    while epoch_idx < args.nepochs:
        epoch_idx += 1

        print('Epoch {}'.format(epoch_idx))
        it = 0

        for x_real, y in train_loader:

            # it += 1


            x_real, y = x_real.to(device), y.to(device)
            z = zdist.sample((batch_size, ))
            # y = clusterer.get_labels(x_real, y).to(device)

            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)


            # Generators updates
            # if args.m:
            #     gloss = trainer.mgenerator_trainstep(y, z)
            # else:
            gloss = trainer.generator_trainstep(y, z)

            if config['training']['take_model_average']:
                update_average(generator_test, generator, beta=config['training']['model_average_beta'])
            # g_scheduler.step()
            # d_scheduler.step()
            # Print stats
            # if it % log_every == 0:
            #     g_loss_last = logger.get_last('losses', 'generator')
            #     d_loss_last = logger.get_last('losses', 'discriminator')
            #     d_reg_last = logger.get_last('losses', 'regularizer')
            #     print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
            #           % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))

            # if it % config['training']['recluster_every'] == 0 and it > config['training']['burnin_time']:
                # print cluster distribution for online methods
            # if it % 100 == 0 and config['training']['recluster_every'] <= 100:
            #     print(f'[epoch {epoch_idx}, it {it}], distribution: {clusterer.get_label_distribution(x_real)}')
            # clusterer.recluster(discriminator=discriminator, x_batch=x_real)

        # (i) Sample if necessary
        if epoch_idx % sample_every == 0 and epoch_idx > 0:
            # print('Creating samples...')
            # os.makedirs(os.path.join(checkpoint_dir[:-7], 'imgs', f'{it}', 'samples'),
            #             exist_ok=True)
            # # for batch in range(10):
            # # x = evaluator.generator(z_sample[batch * 100:(batch + 1) * 100], y_sample[batch * 100:(batch + 1) * 100])
            # x = evaluator.generator(z_sample, y_sample)
            # x = x / 2 + 0.5
            # for j in range(x.shape[0]):
            # #     save_image(copy.deepcopy(x[j]),
            # #                os.path.join( checkpoint_dir[:-7], 'imgs', f'{it}', 'samples',
            # #                             f'%03d.png' % (batch * 100 + j)))
            #     save_image(copy.deepcopy(x[j]),
            #     os.path.join(checkpoint_dir[:-7], 'imgs', f'{it}', 'samples',
            #             f'%03d.png' % j))
            x = evaluator.create_samples(z_test, y_test)
            logger.add_imgs(x, 'all', epoch_idx)
            os.makedirs(path.join(out_dir, 'imgs', f'{epoch_idx}', 'samples'),exist_ok=True)
            with torch.no_grad():####生成0-1SS
                for batch in range(10):
                    x = evaluator.create_samples(z_sample[batch*100:(batch+1)*100], y_sample[batch*100:(batch+1)*100])
                    x = x / 2 + 0.5
                    for j in range(x.shape[0]):
                        save_image(x[j],path.join(out_dir, 'imgs', f'{epoch_idx}', 'samples',
                                            f'%03d.png' %(batch*100+j)))

                # for y_inst in range(sample_nlabels):
                #     x = evaluator.create_samples(z_test, y_inst)
                #     logger.add_imgs(x, '%04d' % y_inst, it)

        # (ii) Compute inception if necessary
        # if epoch_idx % inception_every == 0 :
        #     print('PyTorch Inception score...')
        #     inception_mean, inception_std = evaluator.compute_inception_score()
        #     logger.add('metrics', 'pt_inception_mean', inception_mean, epoch_idx=epoch_idx)
        #     logger.add('metrics', 'pt_inception_stddev', inception_std, epoch_idx=epoch_idx)
        #     print(f'[epoch {epoch_idx}, it {it}] pt_inception_mean: {inception_mean}, pt_inception_stddev: {inception_std}')
        # Print stats
        if epoch_idx % log_every == 0 :
            logger.add('losses', 'discriminator', dloss, epoch_idx=epoch_idx)
            logger.add('losses', 'regularizer', reg, epoch_idx=epoch_idx)
            logger.add('losses', 'generator', gloss, epoch_idx=epoch_idx)

            g_loss_last = logger.get_last('losses', 'generator')
            d_loss_last = logger.get_last('losses', 'discriminator')
            d_reg_last = logger.get_last('losses', 'regularizer')
            print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                  % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))
        # (iii) Backup if necessary
        if epoch_idx % backup_every == 0 :
            # print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % epoch_idx)
            # checkpoint_io.save_clusterer(clusterer, int(it))
            logger.save_stats('stats_%08d.p' % epoch_idx)

        if epoch_idx >= 0:
            checkpoint_io.save('model.pt', it=epoch_idx)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
    parser.add_argument('--nepochs', type=int, default=5000, help='number of epochs to run before terminating')
    parser.add_argument('--model_epoch', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
    parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')
    # parser.add_argument('--m', action='store_true', help='enables margin loss')
    parser.add_argument('--d', type=float, default=0.5)  ###距离边界
    parser.add_argument('--alpha', type=float, default=0.5, help='距离中心')  ##1:没有encircle loss
    parser.add_argument('--beta', type=float, default=0.2, help='dispersion loss weight')  ##1:没有encircle loss
    parser.add_argument('--gamma', type=float, default=0.5, help='fake loss weight')  # gamma
    parser.add_argument('--omega', type=float, default=0.5, help='condition loss weight')
    args = parser.parse_args()
    config = load_config(args.config, 'configs/default.yaml')
    args.outdir = config['training'][
                      'out_dir'] + f'/d{args.d}alpha{args.alpha}beta{args.beta}gamma{args.gamma}omega{args.omega}'
    out_dir = args.outdir

    main()
