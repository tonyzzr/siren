from train_feat_test import *

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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


if __name__ == "__main__":

    # step 0: prepare the original image
    cameraman = ImageFitting(224)
    coords, pixels = cameraman[0]
    original_img_tensor = cameraman.img
    original_img_dataloader = DataLoader(cameraman, 
                                         batch_size=1, 
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=0)

    print("coords.shape: ", coords.shape)
    print("pixels.shape: ", pixels.shape)
    print("original_img_tensor.shape: ", original_img_tensor.shape)

    # step 1: prepare a list of transformations
    # step 2: apply the transformations to the original image
    # to get jittered images -> dataset

    jittered_img_dataset = JitteredImage(original_img_tensor, length = 10)
    jittered_img_dataloader = DataLoader(jittered_img_dataset, batch_size=10, shuffle=True)
    # the returned items are (jittered_img, transform_params)
    # print("jittered_img_dataloader: ", next(iter(jittered_img_dataloader)))
    

    # step 3: prepare a DINO featurizer, a Siren model, and a downsampler
    # and the high-res model input (x, y) in shape of (1, 224*224, 2)
    # and get the high-res feature (model output) in shape of (1, 224*224, feat_dim)
    
    # DINO featurizer
    from featup.featurizers.util import get_featurizer
    dino_backbone, _, dino_feat_dim = get_featurizer('dino16', 
                                   activation_type='token', 
                                   output_root='.')
    dino_backbone.cuda()


    # jittered low-res features ground truth
    # step 4: load images to a dataloader, and randomly
    # sample a batch of 10 jittered images
    from collections import defaultdict

    transform_params_dict = defaultdict(list)
    jittered_feat_list = []

    # step 5: feed the jittered images to the DINO featurizer
    # and get the low-res feature ground truth in shape of
    # (batch_size, feat_dim, 14, 14)
    for jittered_img, transform_params in jittered_img_dataloader:
        jittered_feat_list.append(dino_backbone(jittered_img.cuda()))
        for key, value in transform_params.items():
            transform_params_dict[key].append(value)
    jittered_feat_tensor = torch.cat(jittered_feat_list, dim=0)
    transform_params = {k: torch.cat(v, dim=0) for k, v in transform_params_dict.items()}

    print("jittered_feat_tensor.shape: ", jittered_feat_tensor.shape)
    print("transform_params: ", transform_params)

    # Siren model
    feat_siren = Siren(in_features=2, out_features=dino_feat_dim, 
                      hidden_features=dino_feat_dim, 
                      hidden_layers=3, 
                      outermost_linear=True)
    feat_siren.cuda()
    
    

    total_steps = 2000
    steps_til_summary = 500

    optim = torch.optim.Adam(lr=1e-4, params=feat_siren.parameters())
    model_input, _ = next(iter(original_img_dataloader))
    lr_feat_ground_truth_in_matrix = jittered_feat_tensor.detach()
    print("lr_feat_ground_truth_in_matrix.shape: ", lr_feat_ground_truth_in_matrix.shape)
    # input()

    model_input = model_input.cuda()
    lr_feat_ground_truth_in_matrix = lr_feat_ground_truth_in_matrix.cuda()

    for step in range(total_steps):
        
        model_output, coords = feat_siren(model_input)

        model_output_in_matrix = model_output.contiguous().view(1, 224, 224, dino_feat_dim) # reshape to (b, c, h, w)
        model_output_in_matrix = model_output_in_matrix.permute(0, 3, 1, 2)

        transformed_hr_feats = []
        for idx in range(10):
            selected_tp = {k: v[idx] for k, v in transform_params.items()}
            transformed_hr_feats.append(apply_jitter(model_output_in_matrix, 30, selected_tp))
            # max_pad = 30, temporarily hard-coded

        transformed_hr_feats_in_matrix = torch.cat(transformed_hr_feats, dim=0)
        print("transformed_hr_feats_in_matrix.shape: ", transformed_hr_feats_in_matrix.shape)
        # should be (10, 224, 224, dino_feat_dim)

        pool = nn.AvgPool2d(kernel_size=16)
        predicted_lr_feat_in_matrix = pool(transformed_hr_feats_in_matrix)
        
        print("predicted_lr_feat_in_matrix.shape: ", predicted_lr_feat_in_matrix.shape)
        # should be (10, 384, 14, 14)
        predicted_lr_feat = predicted_lr_feat_in_matrix.view(10, 196, dino_feat_dim)

        print("predicted_lr_feat.shape: ", predicted_lr_feat.shape)
        # should be (10, 196, dino_feat_dim)
        lr_feat_ground_truth = lr_feat_ground_truth_in_matrix.contiguous().view(10, 196, dino_feat_dim)
        print("lr_feat_ground_truth.shape: ", lr_feat_ground_truth.shape)
        # should be (10, 196, dino_feat_dim)

        loss = ((predicted_lr_feat - lr_feat_ground_truth)**2).mean()
        print("loss: ", loss)

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            print()

            plot_feats(original_img_tensor, 
                    pool(model_output_in_matrix)[0, ...], 
                    model_output_in_matrix[0, ...])
            
            plot_feats(original_img_tensor, 
                    lr_feat_ground_truth_in_matrix[0, ...], 
                    pool(transformed_hr_feats_in_matrix)[0, ...])

        optim.zero_grad()
        loss.backward()
        optim.step()


    
    





    # step 6: apply the transformations to the high-res model output
    # and get the high-res feature in shape of (batch_size, 224, 224, feat_dim)


    
    # step 7: downsample the jittered high-res feature to the low-res feature size
    # (batch_size, 14, 14, feat_dim)

    # step 8: calculate the loss between the daonsampled low-res feature and
    #  the ground truth low-res feature

    # step 9: backpropagate the loss and update the parameters of the Siren model

    # step 10: repeat the above steps for a few epochs

    # step 11: save the trained Siren model