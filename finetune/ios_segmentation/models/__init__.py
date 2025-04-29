
def get_model(config):
    if config.get('model_name') == "dgcnn":
        from .dgcnn import DGCNNModule
        model = DGCNNModule(config)
        
    elif config.get('model_name') == "pointnet":
        from .pointnet import PointFirstModule
        model = PointFirstModule(config)
    
    elif config.get('model_name') == "pointtransformer":
        from .point_transformer import PointTransformerModule
        model = PointTransformerModule(config)
        
    elif config.get('model_name') == "ours":
        from .point_transformer import PointTransformerModule
        model = PointTransformerModule(config)
        pretrained_path = 'latest_model.pt'
        new_ckpt = {}
        import torch
        import os
        ckpt = torch.load(os.path.join(config['pretrained'], pretrained_path))
        for key, value in ckpt['model_state_dict'].items():
            if 'module.' in key:
                key = key.replace('module.', '')
            if f'pcd_{config["jaw"]}_encoder' in key:
                new_key = key.replace(f'pcd_{config["jaw"]}_encoder.', 'first_ins_cent_model.')
                new_ckpt[new_key] = value
        a, b = model.load_state_dict(new_ckpt, strict=False)

    elif config.get('model_name') == "tgnet_fps":
        from .tgnet_fps import TGNet_fps
        model = TGNet_fps(config)
        
    elif config.get('model_name') == "tgnet_bdl":
        from .tgnet_bdl import TGNet_bdl
        model = TGNet_bdl(config)
        
    return model