import torch
import torch.nn as nn
import torch.nn.functional as F

from .pt_blocks import TransitionDown, TransitionUp

class PointTransformerEncoder(nn.Module):
    def __init__(self, 
                 block, 
                 blocks, 
                 in_channels, 
                 mask_head=None, 
                 planes=None):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_channels
        self.mask_head = mask_head
        # fdims
        planes = [32, 64, 128, 256, 512]
        # shared head in att
        # if 'share_planes' not in config:
        #     config.share_planes = 8
        # share_planes = config.share_planes

        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(
            block, 
            planes[0], 
            blocks[0], 
            share_planes, 
            stride=stride[0], 
            nsample=nsample[0]
            )  # N/1   - planes(fdims)=32,  blocks=2, nsample=8
        
        self.enc2 = self._make_enc(
            block, 
            planes[1], 
            blocks[1], 
            share_planes, 
            stride=stride[1], 
            nsample=nsample[1]
            )  # N/4   - planes(fdims)=64,  blocks=3, nsample=16
        
        self.enc3 = self._make_enc(
            block, 
            planes[2], 
            blocks[2], 
            share_planes, 
            stride=stride[2], 
            nsample=nsample[2]
            )  # N/16  - planes(fdims)=128, blocks=4, nsample=16KKK
        
        self.enc4 = self._make_enc(
            block, 
            planes[3], 
            blocks[3], 
            share_planes, 
            stride=stride[3], 
            nsample=nsample[3]
            )  # N/64  - planes(fdims)=256, blocks=6, nsample=16
        
        self.enc5 = self._make_enc(
            block, 
            planes[4], 
            blocks[4], 
            share_planes, 
            stride=stride[4], 
            nsample=nsample[4]
            )  # N/256 - planes(fdims)=512, blocks=3, nsample=16
        
        self.dec5 = self._make_dec(block, 
            planes[4], 2, share_planes, nsample=nsample[4], is_head=True
            )  # transform p5
        
        self.dec4 = self._make_dec(block, 
            planes[3], 2, share_planes, nsample=nsample[3] 
            )  # fusion p5 and p4
        
        self.dec3 = self._make_dec(block, 
            planes[2], 2, share_planes, nsample=nsample[2]
            )  # fusion p4 and p3
        
        self.dec2 = self._make_dec(block, 
            planes[1], 2, share_planes, nsample=nsample[1]
            )  # fusion p3 and p2
        
        self.dec1 = self._make_dec(block, 
            planes[0], 2, share_planes, nsample=nsample[0]
            )  # fusion p2 and p1
        

        # self.config = config
        # config.num_layers = block_num
        # config.num_classes = num_classes
        # if 'multi' in config:
        #     self.mask_head = MultiHead(planes, config.multi, config, k=2)
        #     self.cls_head = MultiHead(planes, config.multi, config, k=self.num_classes)
        #     self.offset_head = MultiHead(planes, config.multi, config, k=3)
        # else:
        #     self.cls = nn.Sequential(
        #         nn.Linear(planes[0], planes[0]), 
        #         nn.BatchNorm1d(planes[0]), 
        #         nn.ReLU(inplace=True), 
        #         nn.Linear(planes[0], in_channels))
        #     self.offset_head = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 3))


    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        stride = 1 => TransitionDown = mlp, [block, ...]
        stride > 1 => 
        """
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
            ]
        self.in_planes = planes * block.expansion  # expansion default to 1
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
                )
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        """
        input:
            inputs[0] -> pxo -> batch_size, channel, 24000
            inputs[1] -> target -> batch_size, 24000
            inputs[2] -> pxo_prev -> batch_size, channel(32), 24000
        """
        B, C, N = inputs[0].shape
        pxo = inputs[0].permute(0, 2, 1) # (B, N, C)
        x0 = pxo.reshape(-1, C) # (B * N, C)
        p0 = pxo[:, :, :3].reshape(-1, 3).contiguous() # (B * N, 3)
        o0 = torch.arange(1, B + 1, dtype=torch.int, device=inputs[0].device) # (B, ) (1, 2, ..., B)
        o0 *= N # (B, ) (N, 2N, ..., BN)

        stage_list = {'inputs': inputs}
        
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        down_list = [
            # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
        ]
        # for i, s in enumerate(down_list):
        #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
        stage_list['down'] = down_list

        # x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]  # no upsample - concat with per-cloud mean: mlp[ x, mlp[mean(x)] ]
        # x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        # x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        # x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        # x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        # up_list = [
        #     {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
        #     {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
        #     {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
        #     {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
        #     {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
        # ]
        # stage_list['up'] = up_list

        # if self.cls_head is not None:
        #     cls_results, stage_list = self.cls_head(stage_list)
        #     if B==1:
        #         offset_results, _ = self.offset_head(stage_list)
        #     else:
        #         offset_results = None
        # else:
        #     cls_results = self.cls(x1)
        #     offset_results = self.offset(x1)
        
        # output = []
        # if len(inputs) == 2:
        #     target = inputs[1].reshape(-1) # target: n
        #     target = target.type(torch.long)
        #     target = target + 1 # -1 is gingiva now with out subtraction
        #     info_loss = self.criterion(cls_results, target, stage_list)
        #     output.append(info_loss)

        # cls_results = cls_results.view(B, N, self.num_classes).permute(0,2,1)
        # if B==1:
        #     offset_results = offset_results.view(B, N, 3).permute(0,2,1)
        # else:
        #     offset_results = None
        # output.append(cls_results)
        # output.append(offset_results)
        # output.append(None)
        # output.append(x1)
        
        return down_list


if __name__ == "__main__":    
    # from .pt_blocks import PointTransformerBlock
    # model = PointTransformerEncoder(block=PointTransformerBlock, 
    #                         blocks=[2, 3, 4, 6, 3],
    #                         in_channels=6).cuda()
    # inputs = torch.rand(1, 6, 24000).cuda()
    # targets = torch.randint(0, 12, (1, 1, 24000)).cuda()
    # output = model([inputs, targets])
    # a = 1
    from .pt_blocks import PointTransformerBlock
    model = PointTransformerEncoder(
        block=PointTransformerBlock, 
        blocks=[2, 3, 4, 6, 3],
        in_channels=6
    ).cuda()
    batch_size = 1
    channels = 6
    N = 24000
    points = torch.rand(batch_size, 3, N).cuda()
    features = torch.rand(batch_size, channels, N).cuda()
    offset = torch.arange(1, batch_size + 1, dtype=torch.int32) * N
    print(offset)
    print(points.shape, features.shape, offset.shape)
    output = model(points, features, offset)
    # output = model([features])

    