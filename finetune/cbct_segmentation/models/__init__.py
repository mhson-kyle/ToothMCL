from models.models import *


def get_model(args):
    if args.name == 'ours':
        return Ours(args)
    else:
        print('Without pre-training !')
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
        return model