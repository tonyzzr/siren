from train_feat_test import *

import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, RandomSampler

from collections import defaultdict


def apply_jitter(img, max_pad, transform_params):
    h, w = img.shape[2:]

    padded = F.pad(img, [max_pad] * 4, mode="reflect")

    zoom = transform_params["zoom"].item()
    x = transform_params["x"].item()
    y = transform_params["y"].item()
    flip = transform_params["flip"].item()

    if zoom > 1.0:
        zoomed = F.interpolate(padded, scale_factor=zoom, mode="bilinear")
    else:
        zoomed = padded

    cropped = zoomed[:, :, x:h + x, y:w + y]

    if flip:
        return torch.flip(cropped, [3])
    else:
        return cropped

def sample_transform(use_flips, max_pad, max_zoom, h, w):
    if use_flips:
        flip = random.random() > .5
    else:
        flip = False

    apply_zoom = random.random() > .5
    if apply_zoom:
        zoom = random.random() * (max_zoom - 1) + 1
    else:
        zoom = 1.0

    valid_area_h = (int((h + max_pad * 2) * zoom) - h) + 1
    valid_area_w = (int((w + max_pad * 2) * zoom) - w) + 1

    return {
        "x": torch.tensor(torch.randint(0, valid_area_h, ()).item()),
        "y": torch.tensor(torch.randint(0, valid_area_w, ()).item()),
        "zoom": torch.tensor(zoom),
        "flip": torch.tensor(flip)
    }

class JitteredImage(Dataset):

    def __init__(self, 
                 img, 
                 length = 10, # number of jittered images
                 use_flips = True, 
                 max_zoom = 1.8, 
                 max_pad = 30):
        self.img = img
        self.length = length
        self.use_flips = use_flips
        self.max_zoom = max_zoom
        self.max_pad = max_pad

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        if len(self.img.shape) == 3:
            self.img = self.img.unsqueeze(0)

        h, w = self.img.shape[2:]
        transform_params = sample_transform(self.use_flips, self.max_pad, self.max_zoom, h, w)
        return apply_jitter(self.img, self.max_pad, transform_params).squeeze(0), transform_params

def prepare_lr_feat_ground_truth(dino_backbone,
                                    original_img_tensor,
                                    n_jittered_imgs = 100,
                                    batch_size = 10):

    jittered_img_dataset = JitteredImage(original_img_tensor, length = n_jittered_imgs)
    jittered_img_dataloader = DataLoader(jittered_img_dataset, 
        batch_size=batch_size)
    
    transform_params_dict = defaultdict(list)
    jittered_feat_list = []
    for jittered_img, transform_params in jittered_img_dataloader:
        with torch.no_grad():
            jittered_feat_list.append(dino_backbone(jittered_img.cuda()).cpu())
        for key, value in transform_params.items():
            transform_params_dict[key].append(value)
        del jittered_img, transform_params
        
    jittered_feat_tensor = torch.cat(jittered_feat_list, dim=0)
    transform_params = {k: torch.cat(v, dim=0) for k, v in transform_params_dict.items()}


    lr_feat_ground_truth_in_matrix = jittered_feat_tensor.detach()
    lr_feat_ground_truth_in_matrix = lr_feat_ground_truth_in_matrix
    print("lr_feat_ground_truth_in_matrix.shape: ", lr_feat_ground_truth_in_matrix.shape)

    # lr_feat_ground_truth = lr_feat_ground_truth_in_matrix.contiguous().view(n_jittered_imgs, 196, dino_feat_dim)
    # print("lr_feat_ground_truth.shape: ", lr_feat_ground_truth.shape)

    return lr_feat_ground_truth_in_matrix, transform_params


class ImageFittingColorFeat(Dataset):
    '''
        (x, y, r, g, b) as model input, shape: (1, 224*224, 5)
    '''
    def __init__(self, sidelength):
        super().__init__()
        self.img = get_cameraman_tensor(sidelength)
        print("self.img.shape: ", self.img.shape)
        c, h, w = self.img.shape

        self.pixels = self.img.permute(1, 2, 0).view(-1, c)
        print("self.pixels.shape: ", self.pixels.shape)
        # input()
        self.coords = get_mgrid(sidelength, 2)

        self.combined_input_features = torch.cat([self.coords, self.pixels], dim=1)
        print("self.combined_input_features.shape: ", self.combined_input_features.shape)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels, self.combined_input_features

if __name__ == "__main__":

    # step 0: prepare the original image
    # cameraman = ImageFitting(224)
    cameraman = ImageFittingColorFeat(224)
    coords, pixels, combined_input_features = cameraman[0]
    original_img_tensor = cameraman.img
    original_img_dataloader = DataLoader(cameraman, 
                                         batch_size=1, 
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0)

    
    # DINO featurizer
    from featup.featurizers.util import get_featurizer
    dino_backbone, _, dino_feat_dim = get_featurizer('dino16', 
                                   activation_type='token', 
                                   output_root='.')
    dino_backbone.cuda()
    dino_backbone.eval()

    # prepare the low-res feature ground truths
    n_jittered_imgs = 3000
    batch_size = 10
    all_lr_feat_ground_truth, all_transform_params = prepare_lr_feat_ground_truth(dino_backbone, 
                                                                          original_img_tensor, 
                                                                          n_jittered_imgs = n_jittered_imgs, 
                                                                          batch_size = batch_size)


    # Siren model
    feat_siren = Siren(in_features=combined_input_features.shape[1], 
                       out_features=dino_feat_dim, 
                      hidden_features=dino_feat_dim, 
                      hidden_layers=2, 
                      outermost_linear=True)
    feat_siren.cuda()
    
    

    total_steps = 200
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=feat_siren.parameters())
    _, _, model_input = next(iter(original_img_dataloader))
    
    
    # input()

    model_input = model_input.cuda()

    from featup.downsamplers import SimpleDownsampler
    downsampler = SimpleDownsampler(kernel_size=29, final_size=14)
    downsampler.cuda()

    for step in tqdm(range(total_steps)):
        feat_siren.train()
        downsampler.train()
        
        # get the high-res feature (model output) in shape of (1, 244*244, dino_feat_dim)
        model_output, coords = feat_siren(model_input)
        model_output_in_matrix = model_output.contiguous().view(1, 224, 224, dino_feat_dim) # reshape to (b, c, h, w)
        model_output_in_matrix = model_output_in_matrix.permute(0, 3, 1, 2)

        # prepare the low-res feature ground truths; and apply selected transformations to the high-res feature
        lr_feat_ground_truth_in_matrix_list = []
        transformed_hr_feats_in_matrix_list = []
        for j in range(batch_size):
            idx = torch.randint(n_jittered_imgs, size=())
            lr_feat_ground_truth_in_matrix_list.append(all_lr_feat_ground_truth[idx].unsqueeze(0).cuda())
            selected_tp = {k: v[idx] for k, v in all_transform_params.items()}

            # print("selected_tp: ", selected_tp) # should be different for each batch

            transformed_hr_feats_in_matrix_list.append(apply_jitter(model_output_in_matrix, 30, selected_tp))
            # max_pad = 30, temporarily hard-coded
        lr_feat_ground_truth_in_matrix = torch.cat(lr_feat_ground_truth_in_matrix_list, dim=0)
        transformed_hr_feats_in_matrix = torch.cat(transformed_hr_feats_in_matrix_list, dim=0)
        # print("lr_feat_ground_truth_in_matrix.shape: ", lr_feat_ground_truth_in_matrix.shape)
        # print("transformed_hr_feats_in_matrix.shape: ", transformed_hr_feats_in_matrix.shape)
        # should be (10, dino_feat_dim, 224, 224)

        # pool = nn.AvgPool2d(kernel_size=16)
        # predicted_lr_feat_in_matrix = pool(transformed_hr_feats_in_matrix)

        predicted_lr_feat_in_matrix = downsampler(transformed_hr_feats_in_matrix, None)
        print("predicted_lr_feat_in_matrix.shape: ", predicted_lr_feat_in_matrix.shape)

        loss = ((predicted_lr_feat_in_matrix - lr_feat_ground_truth_in_matrix)**2).mean()
        print("loss: ", loss)

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            print()

            plot_feats(original_img_tensor, 
                    downsampler(model_output_in_matrix, None)[0, ...], 
                    model_output_in_matrix[0, ...])
            
            plot_feats(original_img_tensor, 
                    lr_feat_ground_truth_in_matrix[0, ...], 
                    predicted_lr_feat_in_matrix[0, ...])

        optim.zero_grad()
        loss.backward()
        optim.step()#


    
    





    # step 6: apply the transformations to the high-res model output
    # and get the high-res feature in shape of (batch_size, 224, 224, feat_dim)


    
    # step 7: downsample the jittered high-res feature to the low-res feature size
    # (batch_size, 14, 14, feat_dim)

    # step 8: calculate the loss between the daonsampled low-res feature and
    #  the ground truth low-res feature

    # step 9: backpropagate the loss and update the parameters of the Siren model

    # step 10: repeat the above steps for a few epochs

    # step 11: save the trained Siren model