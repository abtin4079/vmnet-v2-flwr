import torch
from torch.optim import SGD, Adam

# Additional Scripts
from transunet import TransUNet
from utils.utils import dice_loss
from utils.metrics import *

class TransUNetSeg:
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num, device, lr, momentum, weight_decay):
        self.device = device
        self.model = TransUNet(img_dim=img_dim,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               head_num=head_num,
                               mlp_dim=mlp_dim,
                               block_num=block_num,
                               patch_dim=patch_dim,
                               class_num=class_num).to(self.device)

        self.criterion = dice_loss
        self.optimizer = SGD(self.model.parameters(), lr=lr,
                             momentum=momentum, weight_decay=weight_decay)
        # self.optimizer = Adam(self.model.parameters(), lr= lr)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['img'], params['img_sail'])
        loss = self.criterion(pred_mask, params['mask'])
        IOU = intersection_over_union(pred_mask, params['mask'])
        acc = accuracy(pred_mask, params['mask'])
        F1, recall, precision = f1_score(pred_mask, params['mask'])

        loss.backward()

        self.optimizer.step()

        metrics = [IOU , F1 , acc, recall, precision]

        return loss.item(), pred_mask , metrics

    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'], params['img_sail'])
        loss = self.criterion(pred_mask, params['mask'])
        
        a = 0

        return loss.item(), pred_mask , a
