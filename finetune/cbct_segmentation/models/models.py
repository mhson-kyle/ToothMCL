import os

import torch
from monai.networks.nets import SwinUNETR


def Ours(args):
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=True
    )
    if args.feature_size == 48:
        pretrained_path = 'latest_model.pt'
    else:
        print('Error, set args.feature_size in 48, 96, 192')

    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    new_ckpt = {}
    for key, value in model_dict['model_state_dict'].items():
        if 'module.' in key:
            key = key.replace('module.', '')
        if 'image_encoder' in key:
            new_key = key.replace('image_encoder.', '')
            new_ckpt[new_key] = value
    a, b = model.load_state_dict(new_ckpt, strict=False)
    # model = load(model, model_dict)
    print("Using Ours pretrained backbone weights !!!!!!!")
    return model

def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    elif "student" in model_dict.keys():
        state_dict = model_dict["student"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()

    # for k in current_model_dict.keys():
    #     if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
    #         print(k)

    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model



