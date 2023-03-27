import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical_depth(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-2,0,0,0],[0,0,2,0],[0,2,0,0],[0,0,0,2]])) @ c2w
    return c2w

def read_matrix(filepath):
    a = []
    with open(filepath) as f:
        for line in f:
            l = []
            if len(line) > 5:
                ll = line.split('\t')
                l.append(ll[0])
                l.append(ll[1])
                l.append(ll[2])
                l.append(ll[3])
                a.append(l)
    a = np.array(a).astype(np.float32)
    return a



def load_unity_data_testdata(basedir, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,204,skip):
        depth_frame = os.path.join(basedir, str(i)+ 'depth.png')
        mian_fname = os.path.join(basedir, str(i)+ 'main.png')
        matrix_txt = os.path.join(basedir, str(i)+'.txt')

        # depth_frame = os.path.join(basedir, str(i) + 'main.png')
        # mian_fname = os.path.join(basedir, str(i) + 'main.png')
        # matrix_txt = os.path.join(basedir, str(i) + '_pose.txt')
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        # poses.append(np.loadtxt(matrix_txt,delimiter=","))
        poses.append(read_matrix(matrix_txt))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,90),np.arange(150, 170),np.arange(0,204)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    # focal = 1000
    # focal = 600/36*29.088
    focal = 600
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

    depth_imgs = depth_imgs[...,0]
    # depth_imgs[depth_imgs > 0.99] = 0
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split



