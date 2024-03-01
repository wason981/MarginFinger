# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from torchvision import transforms
from torchvision import models
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def build_source_model(dataset):
    victim = models.vgg16_bn(weights=None)
    in_feature = victim.classifier[-1].in_features
    # if dataset=='cifar10':
    #     victim.classifier[-1] = torch.nn.Linear(in_feature, 10)
    #     victim.load_state_dict(torch.load('cifar10/model/Source_Model/source_model.pth'))

    # if dataset=='tiny':
    victim.classifier[-1] = torch.nn.Linear(in_feature, 100)
    victim.load_state_dict(torch.load('model/tiny/Source_Model/source_model.pth'))

    victim = victim.cuda()
    # victim.load_state_dict(torch.load(victim_path))
    # self.victim = nn.DataParallel(self.victim)
    victim.eval()
    return victim

class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 g_optimizer,
                 d_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 c,
                 alpha,
                 beta,
                 gamma,
                 omega,
                 dataset):

        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.c=c
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.omega=omega
        self.dataset=dataset
        self.victim=build_source_model(self.dataset)
        print('D reg gamma', self.reg_param)

    # def generator_trainstep(self, y, z):
    #     assert (y.size(0) == z.size(0))
    #     toggle_grad(self.generator, True)
    #     toggle_grad(self.discriminator, False)
    #
    #     self.generator.train()
    #     self.discriminator.train()
    #     self.g_optimizer.zero_grad()
    #
    #     x_fake = self.generator(z, y)
    #     d_fake = self.discriminator(x_fake, y)
    #     gloss = self.compute_loss(d_fake, 1)
    #     gloss.backward()
    #
    #     self.g_optimizer.step()

        # return gloss.item()
    def generator_trainstep(self, y, z):
        assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()
        # z = torch.load(f'zm10.pt').cuda()
        # y = torch.load(f'ym10.pt').cuda()
        # d_fake= torch.tensor([]).cuda()
        # x_fake= torch.tensor([]).cuda()
        # with torch.no_grad():
        #     for i in range(10):
        #         x_fake_ = self.generator(z[i*100:(i+1)*100], y[i*100:(i+1)*100])
        #         d_fake_ = self.discriminator(x_fake_, y[i*100:(i+1)*100])
        #         d_fake=torch.cat((d_fake,d_fake_),0)
        #         x_fake = torch.cat((x_fake, x_fake_), 0)
        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y=None)
        #########encirclement loss############
        encirclement_loss =self.compute_loss(d_fake, self.alpha)
        #########dispension loss############
        center = torch.mean(x_fake, dim=0, keepdim=True)
        distance_xy = torch.pow(torch.abs(x_fake - center), 2)
        distance = torch.sum(distance_xy, dim=(1,2, 3))
        dispension_loss=torch.pow(torch.mean(distance),-1)
        #########condition loss ##############
        y_pred = torch.softmax(self.victim(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(denorm(x_fake))), dim=1)
        y_pred_clone = y_pred.clone()
        right_pred = y_pred[torch.arange(len(y)), y]
        y_pred_clone[torch.arange(len(y)), y] = -1000
        second_pred = torch.max(y_pred_clone, axis=1).values
        condition_loss = torch.abs(second_pred - right_pred + self.c).mean()
        # condition_loss = torch.relu(second_pred - right_pred + self.c).mean()
        mean=float((right_pred-second_pred).mean().data)
        std=float((right_pred-second_pred).std().data)
        # torch.argmax(y_pred, axis=1)
        gloss = encirclement_loss+ self.beta*dispension_loss +self.omega * condition_loss
        # gloss = self.beta*encirclement_loss +self.omega * condition_loss
        # gloss = 0.5*encirclement_loss+self.omega * condition_loss

        # gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()
    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real,y=None)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y=None)
        dloss_fake =self.gamma*self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(
                x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        dloss = (dloss_real + dloss_fake)
        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
