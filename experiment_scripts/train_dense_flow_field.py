'''
    We will start with a NeRF with (x, y, t) as input and deep feature vector as output.

    The dense flow field of the entire video is also a video with c=2, h=w=224.

    We will compare this dense flow field with the ground truth flow field that generated
    the original video.

'''
import sys, os

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("WORK_DIR: ", WORK_DIR)
sys.path.append(WORK_DIR)

import modules, utils, loss_functions, training

import torch

if __name__ == "__main__":  

    # the feature nerf that we will sample t from
    feat_video_model = modules.SingleBVPNet(type="sine", 
                                            in_features=3, 
                                        out_features=384,
                                        mode='mlp', 
                                        hidden_features=1024, 
                                        num_hidden_layers=3)

    feat_video_model_path = f'{WORK_DIR}/logs/mock_video_feature_test/checkpoints/model_final.pth'
    feat_video_model.load_state_dict(torch.load(feat_video_model_path))
    feat_video_model.eval()
    feat_video_model.cuda()
    # we need this for grid coordinates only
    import dataio
    # feat_video_data_path = f'{WORK_DIR}/data/mock_videos/mockvideo_feat_data.npy'
    # feat_video_dataset = dataio.Video(feat_video_data_path)
    # print("feat_video_dataset.channels: ", feat_video_dataset.channels)
    # print("feat_video_dataset.shape: ", feat_video_dataset.shape)
    # print("--------------------------------")

    # the original video that contains the ground truth flow field
    # from train_featup_video_test_official_dataloader import VideoAndFlowFitting
    # mock_video_data_path = f'{WORK_DIR}/data/mock_videos/translation_h.pt'
    # mockvideo = VideoAndFlowFitting(data_path=mock_video_data_path)

    vid_dataset = dataio.Video('data/mock_videos/mockvideo_vid_data.npy')
    print("vid_dataset.shape: (f, h, w) ", vid_dataset.shape)

    pixel_coords = dataio.get_mgrid(vid_dataset.shape, dim=3)
    print("pixel_coords.shape: ", pixel_coords.shape)

    pixel_coords = pixel_coords.view(vid_dataset.shape[0],
                                        vid_dataset.shape[1],
                                        vid_dataset.shape[2],
                                        3)
    print("pixel_coords.shape: ", pixel_coords.shape)

    dt = pixel_coords[1, 0, 0, 0] - pixel_coords[0, 0, 0, 0]
    dx = pixel_coords[0, 1, 0, 1] - pixel_coords[0, 0, 0, 1]
    dy = pixel_coords[0, 0, 1, 2] - pixel_coords[0, 0, 0, 2]
    print("dt =", dt, "dx =", dx, "dy =", dy)

    t_min, t_max = torch.min(pixel_coords[:, :, :, 0]), torch.max(pixel_coords[:, :, :, 0])
    print("t_min =", t_min, "t_max =", t_max)
    x_min, x_max = torch.min(pixel_coords[:, :, :, 1]), torch.max(pixel_coords[:, :, :, 1])
    y_min, y_max = torch.min(pixel_coords[:, :, :, 2]), torch.max(pixel_coords[:, :, :, 2])
    print("x_min =", x_min, "x_max =", x_max)
    print("y_min =", y_min, "y_max =", y_max)
    # input()
    # ------------------------------------------------------------
    batch_size = 1    
    coord_dataset = dataio.Implicit3DWrapper(vid_dataset, 
                                             sidelength=vid_dataset.shape, 
                                             sample_fraction=38e-4,
                                             )
    
    from torch.utils.data import DataLoader
    coord_dataloader = DataLoader(coord_dataset, 
                            shuffle=True, 
                            batch_size=batch_size,
                            pin_memory=True, 
                            num_workers=0)
    # feat_video_model_input = next(iter(coord_dataloader))


    # print("--------------------------------")
    # print("feat_video_model_input[0].keys(): ", feat_video_model_input[0].keys())
    # print("feat_video_model_input[0]['coords'].shape: ", feat_video_model_input[0]['coords'].shape)
    # print("feat_video_model_input[0]['idx']: ", feat_video_model_input[0]['idx'])
    
    # print("feat_video_model_input[1].keys(): ", feat_video_model_input[1].keys())
    # print("feat_video_model_input[1]['img'].shape: ", feat_video_model_input[1]['img'].shape)
    # print("feat_video_model output: ", feat_video_model(next(iter(coord_dataloader))[0]))
    # input()

    
    
    # ------------------------------------------------------------
    # the dense flow field model
    dense_flow_field_model = modules.SingleBVPNet(type="sine", 
                                                  in_features=3, 
                                                  out_features=2,
                                                  mode='mlp', 
                                                  hidden_features=16, 
                                                  num_hidden_layers=3)

    dense_flow_field_model.cuda()

    num_epochs = 1000
    lr = 1e-4
    steps_til_summary = 100
    lambda_mag = 100

    from train_feat_test import pca
    
    import matplotlib
    # matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    


    optim = torch.optim.Adam(lr=1e-4, params=dense_flow_field_model.parameters())

    for epoch in range(num_epochs):

        loss = 0
        for batch_idx, (model_input, gt) in enumerate(coord_dataloader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            # the model input is a dictionary with two keys: coords and idx
            # idx is always 0, coords is all the (flattened) coordinates in the video
            # [1, f*h*w, 3 (x, y, t)]

            # the gt is a dictionary with one key: img
            # img is all (flattened) the original video
            # [1, f*h*w, c]


            print("--------------------------------")
            print("batch_idx: ", batch_idx)
            print("model_input['idx'].shape: ", model_input['idx'].shape)
            print("model_input['coords'].shape: ", model_input['coords'].shape)
            print("gt['img'].shape: ", gt['img'].shape)

            model_output = dense_flow_field_model(model_input)
            print("model_output['model_out'].shape: ", model_output['model_out'].shape)
            print("model_output['model_in'].shape: ", model_output['model_in'].shape)

            # compare feat_video_model(x0, y0, t0) with feat_video_model(x1, y1, t1)
            # where x1 = x0 + dx, y1 = y0 + dy, t1 = t0 + dt
            # dx, dy = deep_flow_field_model(x0, y0, t0); dt = ()

            coords_source = model_input['coords']
            coords_query = model_input['coords'] + torch.cat([
                model_output['model_out'], # dx, dy
                torch.ones(model_output['model_out'].shape[0], 
                           model_output['model_out'].shape[1],
                           1).cuda() * dt,
            ], dim=-1) 


            feat_video_model_input_source = {
                'coords': coords_source.cuda(),
                'idx': model_input['idx'].cuda(),
            } # this is the same as model_input

            # dx_dy_dt = torch.cat([
            #     model_output['model_out'], # dx, dy
            #     torch.ones(model_output['model_out'].shape[0], 
            #                model_output['model_out'].shape[1],
            #                1).cuda() * dt,
            # ], dim=-1) 

            feat_video_model_input_query = {
                'coords': coords_query.cuda(),
                'idx': model_input['idx'].cuda(),
            }# do we need clamping the max, min value of t, x, t here? 
            # maybe yes, the only problem is that we may extrapolate to the t_max+dt
            # and also x0+dx > x_max, y0+dy > y_max; (or x0+dx < x_min, y0+dy < y_min)
            # extrapolation is suspicious, so we simply clamp the max to avoid problem
            # so if the loss is large in the case of extrapolation, we expect the flow field
            # will adjust to make the deformation smaller to decrease loss
            # but if the loss is small, the flow field will stay at the same and make the 
            # deformed pixel outside the image --------> we need to regularize to minimize the 
            # amount of deformation --------> not clamping but masking + regularization

            roi_mask = torch.ones_like(feat_video_model_input_query['coords'][:, :, 0:1]) * (
                (feat_video_model_input_query['coords'][:, :, 1:2] >= x_min) & 
                (feat_video_model_input_query['coords'][:, :, 1:2] <= x_max) & 
                (feat_video_model_input_query['coords'][:, :, 2:3] >= y_min) & 
                (feat_video_model_input_query['coords'][:, :, 2:3] <= y_max) &
                (feat_video_model_input_query['coords'][:, :, 0:1] >= t_min) &
                (feat_video_model_input_query['coords'][:, :, 0:1] <= t_max)
            ).cuda()
            
            print("roi_mask.shape: ", roi_mask.shape)
            roi_mask = roi_mask.repeat(1, 1, 384)
            print("roi_mask.shape: ", roi_mask.shape)
            # input()
            print("inside ratio: ", roi_mask.sum() / roi_mask.numel())

            feat_video_model_output_source = feat_video_model(feat_video_model_input_source)
            feat_video_model_output_query = feat_video_model(feat_video_model_input_query)


            feat_video_loss = loss_functions.image_mse(roi_mask, feat_video_model_output_query, 
                                            {'img': feat_video_model_output_source['model_out']})
            
            print("feat_video_loss: ", feat_video_loss)

            dense_flow_field_magnitude = torch.sqrt(torch.mean(
                model_output['model_out'] ** 2
            ))
            print("dense_flow_field_magnitude: ", dense_flow_field_magnitude)

            # there are also smoothness regularization, and  out of bound regularization
            # we don't implement them for now, maybe use diffeomorphic flow field later

            batch_loss = feat_video_loss['img_loss'] + dense_flow_field_magnitude**2 * lambda_mag
            print("--------------------------------")
            print("batch_loss: ", batch_loss)
            print("--------------------------------")
            # optim.zero_grad()
            # batch_loss.backward()
            # optim.step()

            loss += batch_loss
        
        loss = loss / len(coord_dataloader)
        print("epoch: ", epoch, "loss: ", loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

        
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        if not epoch % steps_til_summary:
            # input()
            dense_flow_field_model.eval()
            feat_video_model.eval()

            with torch.no_grad():
            # plot the frame 0's, original feature pca, flow field, deformed feature pca
            # plot the frame 1's, original feature pca, error between deformed feature pca and ground truth feature pca
                frame_0_coords = pixel_coords[0, ...].view(1, -1, 3).cuda() # with t
                frame_1_coords = pixel_coords[1, ...].view(1, -1, 3).cuda()

                frame_0_feat_2d = feat_video_model({'coords': frame_0_coords, 'idx': 0})['model_out'].view(1, 224, 224, 384)
                frame_0_flow_field_2d = dense_flow_field_model({'coords': frame_0_coords, 'idx': 0})['model_out'].view(1, 224, 224, 2)
                print("frame_0_flow_field_2d.shape: ", frame_0_flow_field_2d.shape)

                frame_1_feat_2d = feat_video_model({'coords': frame_1_coords, 'idx': 0})['model_out'].view(1, 224, 224, 384)
                # frame_1_flow_field_2d = dense_flow_field_model({'coords': frame_1_coords, 'idx': 0})['model_out'].view(224, 224, 2)

                from prepare_mock_video import visualize_flow
                [frame_0_feat_pca_2d, frame_1_feat_pca_2d], _ = pca([frame_0_feat_2d.permute(0, 3, 1, 2), frame_1_feat_2d.permute(0, 3, 1, 2)])

                print("frame_0_feat_pca_2d.shape: ", frame_0_feat_pca_2d.shape)

                import torch.nn.functional as F
                grid_warpped = torch.cat([pixel_coords[0, :, :, 1:3]]).cuda() + frame_0_flow_field_2d # add flow to the xy coordinates to get the warped grid
                # grid_warpped = grid_warpped[...,::-1] # swap x and y to align coordinate systems    
                warpped_frame_0_feat_pca_2d = F.grid_sample(
                    input = frame_0_feat_pca_2d, # [1, C, H, W]
                    grid = torch.cat([grid_warpped[:, :, :, 1:2], grid_warpped[:, :, :, 0:1]], dim=-1), # [1, H, W, 2] align coordinate systems
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                ) # [1, 384 , 224, 224]

                print("warpped_frame_0_feat_pca_2d.shape: ", warpped_frame_0_feat_pca_2d.shape)
                # input()
                error_2d = torch.abs(frame_1_feat_pca_2d - warpped_frame_0_feat_pca_2d)

            print("************************************************")
            print("start plotting")
            fig, axes = plt.subplots(2, 3)
            axes[0, 0].imshow(frame_0_feat_pca_2d[0, ...].permute(1, 2, 0).cpu().numpy())
            axes[0, 0].set_title("frame 0 original feature pca")

            axes[0, 1].imshow(warpped_frame_0_feat_pca_2d[0, ...].permute(1, 2, 0).cpu().numpy())
            axes[0, 1].set_title("frame 0 deformed feature pca")

            # NOTE THE COLOR OF THE FLOW FIELD IS WRONG FOR NOW
            axes[0, 2].imshow(visualize_flow(frame_0_flow_field_2d[0, ...].permute(2, 0, 1).cpu())) # we may need to swap x and y to align coordinate systems
            axes[0, 2].set_title("frame 0 flow field")

            axes[1, 1].imshow(frame_1_feat_pca_2d[0, ...].permute(1, 2, 0).cpu().numpy())
            axes[1, 1].set_title("frame 1 original feature pca")

            axes[1, 2].imshow(error_2d[0, ...].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            axes[1, 2].set_title("warp error")

            for ax in axes.flatten():
                ax.axis('off')
            
            plt.savefig(f'{WORK_DIR}/logs/dense_flow_field_test/checkpoints/epoch_{epoch}.png')
            plt.show()
            print("************************************************")
            # input()

    



    

    