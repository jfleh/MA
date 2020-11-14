import argparse
import math
import os
import numpy as np
from random import randrange
import matplotlib as pltlib
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import sys
from PIL import Image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from lib.resflow import ResidualFlow
import lib.utils as utils
import lib.layers as layers

import ellipses_dataset as ellipses_ds
from odl.discr.lp_discr import uniform_discr
from odl.tomo import parallel_beam_geometry, RayTransform
from odl.tomo.analytic.filtered_back_projection import fbp_op as fbp
from odl.contrib import torch as odl_torch

import lodopab_small_dataset as lodopab_ds

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, choices=[32, 48, 128], default=32)
parser.add_argument('--angles', type=int, default=30)
parser.add_argument('--noise_stdev', type=float, default=0.01)
parser.add_argument('--freq_scal', type=float, default=1.0)
parser.add_argument('--steps', type=int, default=-1)
parser.add_argument('--stepsize', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lpxweight', type=float, default=0.001) #vllt von stddev abh machen wenn nicht gewaehlt
parser.add_argument('--vis_freq', type=int, default=100)
parser.add_argument('--name', type=str, default='opt')
parser.add_argument('--x0type', type=str, choices=['random', 'fbp', 'both'], default='fbp')
args = parser.parse_args()

pltlib.use('agg')

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

input_size = (32,1, args.size, args.size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#if args.size == 128:
#    nblocks = list(map(int, '8-8-8'.split('-')))
#else:
nblocks = list(map(int, '16-16-16'.split('-')))

model = ResidualFlow(
    input_size,
    n_blocks=nblocks,
    intermediate_dim=512,
    factor_out=False,
    quadratic=False,
    init_layer=layers.LogitTransform(0.05),
    actnorm=True,
    fc_actnorm=False,
    batchnorm=False,
    dropout=0.,
    fc=False,
    coeff=0.98,
    vnorms='2222',
    n_lipschitz_iters=None,
    sn_atol=1e-3,
    sn_rtol=1e-3,
    n_power_series=None,
    n_dist='poisson',
    n_samples=1,
    kernels='3-1-3',
    activation_fn='swish',
    fc_end=True,
    fc_idim=128,
    n_exact_terms=2,
    preact=True,
    neumann_grad=True,
    grad_in_forward=True,
    first_resblock=True,
    learn_p=False,
    classification='density',
    classification_hdim=256,
    n_classes=1,
    block_type='resblock',
)

model.to(device)

with torch.no_grad():
    x = torch.rand(1, *input_size[1:]).to(device)
    model(x)

if args.size == 32:
    if torch.cuda.is_available():
        checkpoint = torch.load("./net32/models/most_recent.pth")
    else:
        checkpoint = torch.load("./net32/models/most_recent.pth",map_location=torch.device('cpu'))
elif args.size == 48:
    if torch.cuda.is_available():
        checkpoint = torch.load("./net48/models/most_recent.pth")
    else:
        checkpoint = torch.load("./net48/models/most_recent.pth",map_location=torch.device('cpu'))
elif args.size == 128:
    if torch.cuda.is_available():
        checkpoint = torch.load("./netlodobig/models/most_recent.pth")
    else:
        checkpoint = torch.load("./netlodobig/models/most_recent.pth",map_location=torch.device('cpu'))

sd = {k: v for k, v in checkpoint['state_dict'].items() if 'last_n_samples' not in k}
state = model.state_dict()
state.update(sd)
model.load_state_dict(state, strict=True)
model.eval()
del checkpoint
del state

if args.size in [32, 48]:
    space = uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=(args.size, args.size), dtype=np.float32)
    geometry = parallel_beam_geometry(space, num_angles=args.angles)
    if torch.cuda.is_available():
        ray_trafo = RayTransform(space, geometry, impl='astra_cuda')
    else:
        ray_trafo = RayTransform(space, geometry, impl='astra_cpu') 
    pair_gen = ellipses_ds.pair_generator(ray_trafo, noise_stddev=args.noise_stdev)
    pair_dataset = ellipses_ds.GeneratorTorchDataset(pair_gen, 32000,shape=((args.angles, -1), (args.size, args.size)))
else:
    lodopab_dival_dataset = lodopab_ds.LoDoPaBSmallDataset(size=args.size, angle_step=20, detector_pixel_step=2)
    num_angles = lodopab_dival_dataset.shape[0][0]
    if torch.cuda.is_available():
        ray_trafo = lodopab_dival_dataset.get_ray_trafo(impl='astra_cuda')
    else:
        ray_trafo = lodopab_dival_dataset.get_ray_trafo(impl='astra_cpu')
    lodopab_dataset = lodopab_dival_dataset.create_torch_dataset(part='test', reshape=((num_angles, -1), (args.size, args.size)))

ray_trafo_fbp = fbp(ray_trafo, filter_type='Hamming', frequency_scaling=args.freq_scal)



def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def get_logpx(x):
    z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
    logpx = logpz -  delta_logp - np.log(256) * args.size * args.size
    return logpx

def momentum_step(y, x0, prev_change = None, stepsize = 0.0005, momentum = 0.9, lpxweight = 0.01):
    x0 = x0.view(args.size,args.size)
    x0 = x0.to(device)
    x0 = torch.clamp(x0, min=0, max=1)
    x0.requires_grad=True
    if (lpxweight == 0):
        prior = 0
        priorgrad = torch.zeros(1).to(device)
    else:
        prior  = lpxweight * get_logpx(x0)
        prior.backward()
        priorgrad = x0.grad
    if args.size == 128:
        fidelity = -poisson_fidelity_term(y, ray_trafo_layer(x0))
        fidelity.backward()
        fidelitygrad = fidelity.grad
    else:
        with torch.no_grad():
            fidelity = -torch.norm(torch.from_numpy(np.asarray((ray_trafo(x0.detach().cpu().numpy())-y))).view(-1), p = 2)
            fidelitygrad = - 2 * ray_trafo.adjoint(ray_trafo(x0.detach().cpu().numpy())-y)
            fidelitygrad = torch.from_numpy(np.asarray(fidelitygrad))
            fidelitygrad = fidelitygrad.to(device)
    with torch.no_grad():
        """ score = []
        score.append(fidelity)
        score.append(lpxweight * logpx0)
        score.append(score[0] + score[1]) """
        score = fidelity + prior
        print('Score: ' + "%.2f" %  score.item() + ' | ' + 'SimGradNorm: ' + "%.2f" % fidelitygrad.view(-1).norm(p = 2).item()
            + ' | ' + 'LogpxGradNorm: ' + "%.2f" % priorgrad.view(-1).norm(p = 2).item())
        if prev_change is None:
            change = fidelitygrad + priorgrad
        else:
            change = momentum * prev_change + fidelitygrad + priorgrad
        x0+= stepsize * change
        x0 = torch.clamp(x0, min=0, max=1)
    return x0, change, (fidelity, prior)

def opt_steps(y, x0, n_steps = 5000, stepsize = 0.0005, momentum = 0.9, lpxweight = 0.01):
    prev = None
    for i in range(n_steps):
        x0, prev, _ = momentum_step(y, x0, prev, stepsize, momentum, lpxweight)
        print('Step ' + str(i+1) + ' of ' + str(n_steps) + ' done.')
    return x0

def log_scores(scores, fidelity, prior, scoreorloss):
        scores[0].append(fidelity)
        scores[1].append(prior)
        scores[2].append(scoreorloss)
        return scores

def vis_opt(y, x, x0, n_steps = 5000, stepsize = 0.0005, momentum = 0.9, lpxweight = 0.01, vis_freq = 100, name = 'opt'):
    # Preparation and logging
    utils.makedirs(os.path.join('opt_images'))
    out = x0.view(-1, *input_size[1:])
    x0save = x0
    xbest = x0
    scores = [[],[],[]]
    highscore = [0, float('-inf'), float('-inf')]
    prev = None
    # Optimization
    if n_steps == -1:
        # Keep going until there is no more improvement
        current = 0
        while(current - highscore[0] < 1000):
            xbackup = x0
            x0, prev, (fidelity, prior) = momentum_step(y, x0, prev, stepsize, momentum, lpxweight)
            score = fidelity + prior
            scores = log_scores(scores, fidelity, prior, score)
            # Update highscore
            if (score > highscore[2]):
                xbest = xbackup
                highscore[0] = current
                highscore[1] = prior
                highscore[2] = score
            print('Step ' + str(current+1) + ' done. Best score ' + "%.2f" % highscore[2].item() + ' after step ' + str(highscore[0]) + '.')
            if ((current+1) % vis_freq == 0) or ((current+1) - highscore[0] >= 1000):
                out = torch.cat([out, x0.cpu().view(-1, *input_size[1:])], 0)
            current+=1
    else:
        # Fixed number of steps
        for i in range(n_steps):
            xbackup = x0
            x0, prev, (fidelity, prior) = momentum_step(y, x0, prev, stepsize, momentum, lpxweight)
            score = fidelity + prior
            scores = log_scores(scores, fidelity, prior, score)
            if (score > highscore[2]):
                xbest = xbackup
                highscore[0] = i
                highscore[1] = prior
                highscore[2] = score
            print('Step ' + str(i+1) + ' of ' + str(n_steps) + ' done. Best score ' + "%.2f" % highscore[2].item() + ' after step ' + str(highscore[0]) + '.')
            if ((i+1) % vis_freq == 0) or (i+1 == n_steps):
                out = torch.cat([out, x0.cpu().view(-1, *input_size[1:])], 0)
    # Calculate final Score for the plot
    with torch.no_grad():
        prior  = lpxweight * get_logpx(x0)
        if args.size == 128:
            fidelity = -poisson_fidelity_term(y, ray_trafo_layer(x0))
        else:
            fidelity = -torch.norm(torch.from_numpy(np.asarray((ray_trafo(x0.detach().cpu().numpy())-y))).view(-1), p = 2)
        score = fidelity + prior
        scores = log_scores(scores, fidelity, prior, score)
        if (score > highscore[2]):
            xbest = x0
    out = torch.cat([out,  xbest.cpu().view(-1, *input_size[1:]), x.view(-1, *input_size[1:]), x0save.cpu().view(-1, *input_size[1:])], 0)
    plot_stats(xbest, x, x0save, out, stepsize, lpxweight, scores, highscore, name, momentum)
"""     if args.size == 32:
        rows = 16
    elif args.size == 48:
        rows = 11
    elif args.size == 128:
        rows = 8
    save_image(out.cpu().float(), os.path.join('opt_images', name + '.png'), nrow=rows, padding=2)
    # Plot
    if n_steps == -1:
        n_steps = current
    if args.size in [32, 48]:
        plt.rcParams.update({'font.size': 6})
    else:
        plt.rcParams.update({'font.size': 8})
    t = np.arange(0, n_steps+1, 1)
    fig, ax = plt.subplots()
    if args.size == 32:
        fig.set_size_inches(5.44, 4.08)
    elif args.size == 48:
        fig.set_size_inches(5.52, 4.14)
    elif args.size == 128:
        fig.set_size_inches(10.88, 8.16)
        #fix 2px spacing
    ax.plot(t, scores[0], 'g')
    ax.plot(t, scores[1], 'r')
    ax.plot(t, scores[2], 'b')
    titlestr = 'Steps: ' + str(n_steps) + ' | Stepsize: ' + str(stepsize) + ' | Momentum: ' + str(momentum) + ' | LogPx weight: ' + str(lpxweight) + '\nNoise: ' + str(args.noise_stdev) +  ' | Angles: ' + str(args.angles)
    if args.x0type in ['fbp', 'both']:
        titlestr = titlestr + ' | Frequency Scaling: ' + str(args.freq_scal)
    titlestr = titlestr + '\nPSNR: ' + "%.2f" % x0psnr.item() + '/' + "%.2f" % xbestpsnr.item() + ' | SSIM: ' +  "%.4f" % x0ssim.item() + '/' + "%.4f" % xbestssim.item()
    if args.lpxweight != 0:
        titlestr = titlestr + ' | LogP: ' + "%.0f" % lgpx0.item() + '/' + "%.0f" % lgpxbest.item() + '/' + "%.0f" % lgpx.item()
    ax.set(frame_on = False, xlabel = 'Steps', ylabel = 'Score', title = titlestr)
    ax.grid()
    fig.savefig(os.path.join('opt_images', name + '-plot.png'))
    # Join images and plot
    img = Image.open(os.path.join('opt_images', name + '.png'))
    imgplot = Image.open(os.path.join('opt_images', name + '-plot.png'))
    join_img_vert([img, imgplot], os.path.join('opt_images', name + '+plot.png')) """

def plot_stats(xbest, x, x0, out, stepsize, lpxweight, scores, highscore, name, momentum = None):
    xbest = xbest.view(args.size, args.size)
    x = x.view(args.size, args.size)
    x0 = x0.view(args.size, args.size)
    x0psnr = psnr(x.numpy(), x0.numpy())
    x0ssim = ssim(x.numpy(), x0.numpy())
    xbestpsnr = psnr(x.numpy(),xbest.detach().cpu().numpy())
    xbestssim = ssim(x.numpy(),xbest.detach().cpu().numpy())
    if (lpxweight != 0):
        lgpx0 = scores[1][0]/lpxweight
        lgpxbest = highscore[1]/lpxweight
        lgpx = get_logpx(x.to(device))
    if args.size == 32:
        rows = 16
    elif args.size == 48:
        rows = 11
    elif args.size == 128:
        rows = 8
    save_image(out.cpu().float(), os.path.join('opt_images', name + '.png'), nrow=rows, padding=2)
    # Plot
    n_steps = len(scores[0])
    if args.size in [32, 48]:
        plt.rcParams.update({'font.size': 6})
    else:
        plt.rcParams.update({'font.size': 8})
    t = np.arange(0, n_steps, 1)
    fig, ax = plt.subplots()
    if args.size == 32:
        fig.set_size_inches(5.44, 4.08)
    elif args.size == 48:
        fig.set_size_inches(5.52, 4.14)
    elif args.size == 128:
        fig.set_size_inches(10.88, 8.16)
        #fix 2px spacing TODO
    ax.plot(t, scores[0], 'g')
    ax.plot(t, scores[1], 'r')
    ax.plot(t, scores[2], 'b')
    titlestr = 'Steps: ' + str(n_steps) + ' | Stepsize: ' + str(stepsize)
    if momentum is not None:
        titlestr += ' | Momentum: ' + str(momentum)
    titlestr += ' | LogPx weight: ' + str(lpxweight) + '\nNoise: ' + str(args.noise_stdev) +  ' | Angles: ' + str(args.angles)
    if args.x0type in ['fbp', 'both']:
        titlestr += ' | Frequency Scaling: ' + str(args.freq_scal)
    titlestr += '\nPSNR: ' + "%.2f" % x0psnr.item() + '/' + "%.2f" % xbestpsnr.item() + ' | SSIM: ' +  "%.4f" % x0ssim.item() + '/' + "%.4f" % xbestssim.item()
    if args.lpxweight != 0:
        titlestr += ' | LogP: ' + "%.0f" % lgpx0.item() + '/' + "%.0f" % lgpxbest.item() + '/' + "%.0f" % lgpx.item()
    ax.set(frame_on = False, xlabel = 'Steps', ylabel = 'Score', title = titlestr)
    ax.grid()
    fig.savefig(os.path.join('opt_images', name + '-plot.png'))
    # Join images and plot
    img = Image.open(os.path.join('opt_images', name + '.png'))
    imgplot = Image.open(os.path.join('opt_images', name + '-plot.png'))
    join_img_vert([img, imgplot], os.path.join('opt_images', name + '+plot.png'))


def torch_opt(y, x, x0, n_steps = 5000, learningrate = 0.00001, lpxweight = 0.1, vis_freq = 100, name = 'torch_opt'):
    # Preparation and logging
    utils.makedirs(os.path.join('opt_images'))
    out = x0.view(-1, *input_size[1:])
    x0save = x0
    xbest = x0
    scores = [[],[],[]]
    lowscore = [0, float('inf'), float('inf')]
    # Optimization
    x0 = x0.view(1,1, args.size,args.size)
    x0 = x0.to(device)
    x0 = torch.clamp(x0, min=0, max=1)
    x0.requires_grad=True
    ray_trafo_layer = odl_torch.operator.OperatorModule(ray_trafo).cuda()
    #fbp_layer = odl_torch.operator.OperatorModule(ray_trafo_fbp).cuda()
    x0_optimizer = torch.optim.Adam([x0], lr=learningrate)
    if n_steps == -1:
        # Keep going until there is no more improvement
        current = 0
        while(current - lowscore[0] < 1000):
            xbackup = x0
            x0_optimizer.zero_grad()
            if (lpxweight == 0):
                prior = 0
            else:
                prior  = lpxweight*get_logpx(x0)
            fidelity = poisson_fidelity_term(y, ray_trafo_layer(x0))
            varloss = fidelity - prior
            with torch.no_grad():
                scores = log_scores(scores, fidelity, prior, varloss)
                # Update highscore
                if (varloss < lowscore[2]):
                    xbest = xbackup
                    lowscore[0] = current
                    lowscore[1] = prior
                    lowscore[2] = varloss
            varloss.backward()
            #print(x0.grad)
            x0_optimizer.step()
            with torch.no_grad():
                x0.clamp_(min=0, max=1)
                print('Step ' + str(current+1) + ' done. Best score ' + "%.2f" % lowscore[2].item() + ' after step ' + str(lowscore[0]) + '.')
                if ((current+1) % vis_freq == 0) or ((current+1) - lowscore[0] >= 1000):
                    out = torch.cat([out, x0.cpu().view(-1, *input_size[1:])], 0)
            current += 1
    else:
        # Fixed number of steps
        for step in range(n_steps):
            xbackup = x0
            x0_optimizer.zero_grad()
            if (lpxweight == 0):
                prior = 0
            else:
                prior  = lpxweight*get_logpx(x0)
            fidelity = poisson_fidelity_term(y, ray_trafo_layer(x0))
            varloss = fidelity - prior
            with torch.no_grad():
                scores = log_scores(scores, fidelity, prior, varloss)
                # Update highscore
                if (varloss < lowscore[2]):
                    xbest = xbackup
                    lowscore[0] = step
                    lowscore[1] = prior
                    lowscore[2] = varloss
            varloss.backward()
            #print(x0.grad)
            x0_optimizer.step()
            with torch.no_grad():
                x0.clamp_(min=0, max=1)
                print('Step ' + str(step+1) + ' done. Best score ' + "%.2f" % lowscore[2].item() + ' after step ' + str(lowscore[0]) + '.')
                if ((step+1) % vis_freq == 0) or ((step+1) == n_steps):
                    out = torch.cat([out, x0.cpu().view(-1, *input_size[1:])], 0)
    out = torch.cat([out,  xbest.cpu().view(-1, *input_size[1:]), x.view(-1, *input_size[1:]), x0save.cpu().view(-1, *input_size[1:])], 0)
    plot_stats(xbest, x, x0save, out, learningrate, lpxweight, scores, lowscore, name)

#def plot

def poisson_fidelity_term(y, Ax,
                          PHOTONS_PER_PIXEL = 4096,
                          MU_WATER = 20, MU_AIR = 0.02):
    """
    This is a fidelity term for the LoDoPaB dataset based on the KL-Divergence.
    This can be motivated via the creation of the dataset, see
    `https://github.com/jleuschn/lodopab_tech_ref/blob/master/create_dataset.py`,
    and the Bayesian approach to inverse problems, i.e., the likelihood of y, given yδ.
    """

    def get_photon_version(y, ε=1e-5):
        MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

        y = y * - MU_MAX
        y = torch.exp(y)
        y = y * PHOTONS_PER_PIXEL

        y[y < 0.1 / PHOTONS_PER_PIXEL + ε] = 0
        return y

    Ax =  Ax.to(device)
    y =  y.to(device)

    Ax = get_photon_version(Ax)
    y = get_photon_version(y)

    Ax = torch.clamp(Ax, 0.1 / PHOTONS_PER_PIXEL, np.inf)

    return torch.sum(Ax - y*torch.log(Ax)) / Ax.numel()


def join_img_vert(images, saveaspath):
    # Takes a list of opened images and joins them vertically
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]

    new_im.save(saveaspath)

def main():
    if args.size in [32, 48]:
        (y,x) = pair_dataset[None]
        if args.x0type  in ['fbp', 'both']:
            x0 = ray_trafo_fbp(y)
            x0 = torch.from_numpy(np.asarray(x0))
            vis_opt(y, x, x0, n_steps = args.steps, stepsize = args.stepsize, momentum = args.momentum,
                lpxweight = args.lpxweight, vis_freq = args.vis_freq, name = (args.name + '-fbp'))
        if args.x0type  in ['random', 'both']:
            x0 = torch.rand(args.size,args.size)
            vis_opt(y, x, x0, n_steps = args.steps, stepsize = args.stepsize, momentum = args.momentum,
                lpxweight = args.lpxweight, vis_freq = args.vis_freq, name = (args.name + '-random'))
    else:
        (y,x) = lodopab_dataset[randrange(3554)]
        if args.x0type  in ['fbp', 'both']:
            x0 = ray_trafo_fbp(y)
            x0 = torch.from_numpy(np.asarray(x0))
            torch_opt(y, x, x0, n_steps = args.steps, learningrate = args.stepsize, lpxweight = args.lpxweight, name = (args.name + '-fbp'))
        if args.x0type  in ['random', 'both']:
            x0 = torch.rand(args.size,args.size)
            torch_opt(y, x, x0, n_steps = args.steps, learningrate = args.stepsize, lpxweight = args.lpxweight, name = (args.name + '-random'))
        
if __name__ == '__main__':
    main()
