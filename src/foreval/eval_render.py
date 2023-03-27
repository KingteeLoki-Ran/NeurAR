import os, sys
import numpy as np
import imageio
import json
import random
import time
# from load_blender import load_blender_data_depth
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import copy

import matplotlib.pyplot as plt

from foreval.eval_helpers import *

from skimage.metrics import structural_similarity as ssim
import lpips
import cv2 as cv

# from load_llff import load_llff_data
# from load_deepvoxels import load_dv_data
# from load_blender import load_blender_data
# from load_LINEMOD import load_LINEMOD_data

import logging


device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

loss_fn_alex = lpips.LPIPS(net='squeeze').to(device)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        logging.debug("applying fn and inputs length is  " + str(inputs.shape) + " , the size of chunk is " + str(chunk))
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]).to(device), far * torch.ones_like(rays_d[...,:1]).to(device)
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncertainty_map','depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_list.append(rays_o)
    ret_list.append(rays_d)
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(gt,start,basedir,expname,testdata,render_poses,offset, hwf, K, chunk, render_kwargs, gt_imgs=None,depth_imgs=None,index = None, far = None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    uncers = []
    depths = []

    allpoints = torch.zeros((0,3)).to(device)
    allpoints_gt = torch.zeros((0,3)).to(device)

    testsavedir = os.path.join(basedir, expname,'atestdata',testdata + '_{:06d}'.format(start))
    os.makedirs(testsavedir, exist_ok=True)

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, uncer, depth, rays_o, rays_d, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        

        target_depth_un = depth_imgs[index[i]]
        img_loss_perray_un = img2mesrespective(rgb, gt_imgs[index[i]])
        img_loss_un = torch.sum(img_loss_perray_un)
        delta_un = torch.sum(uncer)
        depth_diff_un = target_depth_un - (depth/far)
        depth_loss_un = torch.sum(depth_diff_un**2)
        ####################################################
        depth_diff_un = torch.abs(depth_diff_un)
        depth_calc = depth_diff_un[target_depth_un != 1]
        depth_avg = torch.sum(depth_calc)/len(depth_calc)
        depth_avg = depth_avg*far
        # ratio3 = len(depth_calc[depth_calc<(0.03/6)])/len(depth_calc)

        loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un

        psnr_un = mse2psnr(img2msepsnr(rgb, gt_imgs[index[i]]))
        

        in1  = rgb.T[None,...]
        in2  = gt_imgs[index[i]].T[None,...]
        lpips_value = loss_fn_alex(in1, in2)

        if savedir is not None:
            # print("begin rgb")
            rgb8 = to8b(rgb.cpu().numpy())
            filename = os.path.join(savedir, '{:03d}.png'.format(i+offset))
            imageio.imwrite(filename, rgb8)
        
        ssim_value = ssim(rgb.cpu().numpy(), gt_imgs[index[i]].cpu().numpy(),data_range = 1,channel_axis=2)

        ####################################### sample surface points ###############################################
        depth = depth.reshape((H*W),1)

        uncer_data = torch.sum(uncer,-1)
        uncer_data = uncer_data.reshape((H,W))
        uncer_data = 1/uncer_data
        uncer_data = uncer_data.cpu().numpy()
        rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
        filename_img = os.path.join(testsavedir, 'uncer_{:03d}.png'.format(i + offset))
        imageio.imwrite(filename_img, rgb8)
        filename = os.path.join(basedir, expname,'atestdata', testdata+'.txt')
        with open(filename,'a') as f: 
            f.write("current step is + " + str(start)+"\n")
            f.write("current img is + " + str(index[i]) +"\n")
            f.write("img loss is + " + str(img_loss_un.item()) +"\n")
            f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
            f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
            f.write("final loss is + " + str(loss_un.item()) +"\n")
            f.write("psnr is + " + str(psnr_un.item()) +"\n")
            f.write("ssim is + " + str(ssim_value)+"\n")
            f.write("lpips is + " + str(lpips_value.item())+"\n")
            f.write("depth average is + " + str(depth_avg.item())+"\n")
            # f.write("ratio3 is + " + str(ratio3)+"\n")


        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        


    # rgbs = np.stack(rgbs, 0)
    # disps = np.stack(disps, 0)

    return rgbs, disps, uncers, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    logging.info("corase model grad_vars length :  " + str(len(grad_vars)))

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    logging.info("both model grad_vars length :  " + str(len(grad_vars)))

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)



    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    logging.info("Load checkpoints")
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logging.info('Found ckpts' + str(ckpts))
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logging.info('Reloading from' + str(ckpt_path))
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        logging.info("check point : " + str(ckpt))
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    logging.info("render_kwargs_train : " + str(render_kwargs_train) + " , render_kwargs_test : " + str(render_kwargs_test) + " , start : " + str(start))
    logging.info("grad_vars lengths: " + str(len(grad_vars)))
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(mask, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    ############################### draw path and views ##############################
    # color = torch.Tensor([0,0,0]).to(device)
    # density = torch.Tensor([20]).to(device)
    # rgb[mask] = color
    # raw[...,3][mask] = density
    ############################### draw path and views ##############################
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def get_pose(u,v):
    # u = np.pi * u / 180
    # v = np.pi * v / 180
    sx = np.sin(u)
    cx = np.cos(u)
    sy = np.sin(v)
    cy = np.cos(v)
    return torch.Tensor([[cy, sy*sx, -sy*cx],
                        [0, cx, sx],
                        [-sy, cy*sx, -cy*cx]]).to(device)


def render_rays(
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]).to(device)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1).to(device)
        lower = torch.cat([z_vals[...,:1], mids], -1).to(device)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn)

    mask = pts != None

    
 
    uncertainty_map = raw[...,4]
    logging.debug("get from coarse model raw's shape : " + str(raw.shape) + " uncertainty shape :" + str(uncertainty_map.shape))
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mask,raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, uncertainty_map0, depth_map0 = rgb_map, disp_map, acc_map, uncertainty_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        
        
        

        mask = pts != None
       

        uncertainty_map = raw[...,4]
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mask, raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        logging.debug("get from fine model raw's shape : " + str(raw.shape) + " uncertainty shape :" + str(uncertainty_map.shape))
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'uncertainty_map' : uncertainty_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['uncertainty_map0'] = uncertainty_map0
        ret['depth_map0'] = depth_map0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



