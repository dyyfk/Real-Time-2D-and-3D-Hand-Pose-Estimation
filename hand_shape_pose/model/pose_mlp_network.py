from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_shape_pose.model.net_hg import Net_HM_HG
from hand_shape_pose.model.net_mlp import MLP
from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.graph_util import build_hand_graph
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image, uvd2xyz, uvd2xyz_freihand_train, uvd2xyz_freihand_test
from hand_shape_pose.util.heatmap_util import compute_uv_from_heatmaps



### MODIFICATION ###


import matplotlib.pyplot as plt

def show(*imgs):
    '''
     input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
     Single input example:
     show(x) gives the visualization of x, where x should be a torch.Tensor
        if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most. 
        if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
        if x is a 2D tensor, it will be shown as grayscale map
     
     Multiple input example:      
     show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
    '''
    img_idx = 0
    for img in imgs:
        img_idx +=1
        plt.figure(img_idx)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            if img.dim()==4: # 4D tensor
                bz = img.shape[0]
                c = img.shape[1]
                if bz==1 and c==1:  # single grayscale image
                    img=img.squeeze()
                elif bz==1 and c==3: # single RGB image
                    img=img.squeeze()
                    img=img.permute(1,2,0)
                elif bz==1 and c > 3: # multiple feature maps
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
                elif bz > 1 and c == 1:  # multiple grayscale images
                    img=img.squeeze()
                elif bz > 1 and c == 3:  # multiple RGB images
                    img = img.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # multiple feature maps
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim()==3: # 3D tensor
                bz = 1
                c = img.shape[0]
                if c == 1:  # grayscale
                    img=img.squeeze()
                elif c == 3:  # RGB
                    img = img.permute(1, 2, 0)
                else:
                    raise Exception("unsupported type!  " + str(img.size()))
            elif img.dim()==2:
                pass
            else:
                raise Exception("unsupported type!  "+str(img.size()))


            img = img.numpy()  # convert to numpy
            img = img.squeeze()
            if bz ==1:
                plt.imshow(img, cmap='gray')
                # plt.colorbar()
                # plt.show()
            else:
                for idx in range(0,bz):
                    plt.subplot(int(bz**0.5),int(np.ceil(bz/int(bz**0.5))),int(idx+1))
                    plt.imshow(img[idx], cmap='gray')

        else:
            raise Exception("unsupported type:  "+str(type(img)))

### END OF MODIFICATION ### 


class MLPPoseNetwork(nn.Module):
    """
    Main class for 3D hand shape and pose inference network.
    It consists of three main parts:
    - heat-map estimation network
    - mlp pose estimation network
    """
    def __init__(self, cfg):
        super(MLPPoseNetwork, self).__init__()
        """
        # 1. Build graph for Hand Mesh
        self.graph_L, self.graph_mask, self.graph_perm_reverse, self.hand_tri = \
            build_hand_graph(cfg.GRAPH.TEMPLATE_PATH, output_dir)
        """
        # 2. Create model
        num_joints = cfg.MODEL.NUM_JOINTS
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
                                num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
                                num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
        self.device = cfg.MODEL.DEVICE

        num_heatmap_chan = self.net_hm.numOutput
        num_feat_chan = self.net_hm.nFeats

        self.mlp = MLP(num_heatmap_chan, num_feat_chan)
        """
        num_mesh_output_chan = 3
        num_pose_output_chan = (num_joints - 1)# * 3

        self.net_feat_mesh = Net_HM_Feat_Mesh(num_heatmap_chan, num_feat_chan,
                                              num_mesh_output_chan, self.graph_L)
        self.net_mesh_pose = Graph_CNN_Mesh_Pose(num_mesh_output_chan,
                                                 num_pose_output_chan, self.graph_L)
        """
    def load_model(self, cfg, load_mlp=False):
        load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, self.net_hm)
        if load_mlp:
            load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.MLP_NET_PATH, self.mlp)

        """
        load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.MESH_NET_PATH, self.net_feat_mesh)
        load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.POSE_NET_PATH, self.net_mesh_pose)
        """

    def to(self, *args, **kwargs):
        super(MLPPoseNetwork, self).to(*args, **kwargs)
        #self.graph_L = [l.to(*args, **kwargs) for l in self.graph_L]
        #self.net_feat_mesh.mesh_net.graph_L = self.graph_L
        #self.net_mesh_pose.graph_L = self.graph_L
        #self.graph_mask = self.graph_mask.to(*args, **kwargs)

    def forward(self, images, cam_param, pose_scale, pose_root=None, bbox=None):
        """
        :param images: B x H x W x C
        :param cam_param: B x 4, [fx, fy, u0, v0]
        :param bbox: B x 4, bounding box in the original image, [x, y, w, h]
        :param pose_root: B x 3
        :param pose_scale: B
        :return:
        """
        num_sample = images.shape[0]
        if pose_root is not None:
            root_depth = pose_root[:, -1]
            
        show(images[0])
        images = BHWC_to_BCHW(images)  # B x C x H x W
        images = normalize_image(images)
        #### MODIFICATION #### 
        
#         print(images)
#         image = images[0]
#         plt.imshow(image.permute(1, 2, 0).detach().to("cpu").numpy())

#         image = image_paths[random.randint(0, 1000)]
#         original_image = mpimg.imread(image)
#         translated_image = pan(original_image)

#         fig, axes =plt.subplots(1,2,figsize=(12,10))
#         fig.tight_layout()

#         axes[0].imshow(original_image)
#         axes[0].set_title('Original image')
#         axes[1].imshow(translated_image)
#         axes[1].set_title('Translated image')
        
        
        #### END OF MODIFICATION ####
        

        # 1. Heat-map estimation
        est_hm_list, encoding = self.net_hm(images)

        # Pose Estimation
        est_pose_rel_depth = self.mlp(est_hm_list, encoding) # This relative depth is scale invariant
        est_pose_rel_depth.unsqueeze_(-1)

        # combine heat-map estimation results to compute pose xyz in camera coordiante system
        est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], (224, 224))  # B x K x 3
        #if gt_uv is None:
        est_pose_uvd = torch.cat((est_pose_uv[:, 1:, :2],
                                  est_pose_rel_depth[:, :, -1].unsqueeze(-1)), -1)  # B x (K-1) x 3
        #else:
        #    est_pose_uvd = torch.cat((gt_uv[:, 1:, :],
        #                              est_pose_rel_depth[:, :, -1].unsqueeze(-1)), -1)  # B x (K-1) x 3

        root_uv = est_pose_uv[:, 0, :2]
        if bbox is not None: # STB dataset or real world testset
            est_pose_uvd[:, :, 0] = est_pose_uvd[:, :, 0] / float(images.shape[2])
            est_pose_uvd[:, :, 1] = est_pose_uvd[:, :, 1] / float(images.shape[3])
        if bbox is None: # FreiHAND dataset
            if pose_root is None: # FreiHAND testset
                # When root pose is not given, perform 2.5D pose regression to reconstruct normalized pose including the root joint
                est_pose_cam_xyz_with_root = uvd2xyz_freihand_test(est_pose_uvd, root_uv, cam_param, pose_scale)  # B x K x 3
                return est_hm_list[-1], est_pose_uv[:, :, :2], est_pose_cam_xyz_with_root
            else: # FreiHAND trainset
                est_pose_cam_xyz = uvd2xyz_freihand_train(est_pose_uvd, cam_param, root_depth, pose_scale)  # B x (K-1) x 3
                est_pose_cam_xyz = torch.cat((pose_root.unsqueeze(1), est_pose_cam_xyz.to(self.device)), 1)  # B x K x 3
        else: # STB dataset or real world testset
            est_pose_cam_xyz = uvd2xyz(est_pose_uvd, cam_param, bbox, root_depth, pose_scale)  # B x (K-1) x 3
            est_pose_cam_xyz = torch.cat((pose_root.unsqueeze(1), est_pose_cam_xyz), 1)  # B x K x 3

        return est_hm_list[-1], est_pose_uv[:, :, :2], est_pose_cam_xyz
