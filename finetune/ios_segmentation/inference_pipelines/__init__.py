import torch

def make_inference_pipeline(model_name, ckpt_path_ls):
    inference_config = {
        "model_info":{
            "model_parameter" :{
                "input_feat": 6,
                "stride": [1, 4, 4, 4, 4],
                "nstride": [2, 2, 2, 2],
                "nsample": [36, 24, 24, 24, 24],
                "blocks": [2, 3, 4, 6, 3],
                "block_num": 5,
                "planes": [32, 64, 128, 256, 512],
                "crop_sample_size": 3072,
            },
        },
    }
    from inference_pipelines.inference_pipeline_sem import InferencePipeLine
    from models.point_transformer import PointTransformerModule
    module = PointTransformerModule(inference_config["model_info"])
    checkpoint = torch.load(ckpt_path_ls)
    module.load_state_dict(checkpoint['model_state_dict'])
    module.cuda()
    return InferencePipeLine(module)