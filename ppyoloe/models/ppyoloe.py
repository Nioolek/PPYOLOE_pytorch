import torch.nn as nn


class PPYOLOE(nn.Module):

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, targets=None, extra_info=None):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)

        if self.training:
            assert (targets is not None) and extra_info is not None
            yolo_losses = self.head(neck_feats, targets, extra_info)
            return yolo_losses
        else:
            outputs = self.head(neck_feats)
            return outputs

