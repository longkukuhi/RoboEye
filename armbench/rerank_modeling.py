import torch
import torch.nn as nn
from timm.models.registry import register_model
from vggt_adapter.models.vggt import VGGT_adapter
from timm.models.layers import trunc_normal_


class VGGTforRerank(nn.Module):
    def __init__(self, **kwargs):
        super(VGGTforRerank, self).__init__(**kwargs)
        self.vggt = VGGT_adapter()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_points, query_image, positive_image, negative_image0, negative_image1, negative_image2):
        B = query_image.size(0)
        positive = torch.cat([query_image, positive_image], dim=1)  #[B, 2, C, H, W]
        #pos_predictions = self.vggt.forward(positive, query_points) 
        #pos_conf = pos_predictions["conf"]  #[B, 2, N]
        #pos_conf = torch.mean(pos_conf[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative0 = torch.cat([query_image, negative_image0], dim=1)  #[B, 2, C, H, W]
        #neg_predictions0 = self.vggt.forward(negative0, query_points)
        #neg_conf0 = neg_predictions0["conf"]  #[B, 2, N]
        #neg_conf0 = torch.mean(neg_conf0[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative1 = torch.cat([query_image, negative_image1], dim=1)  #[B, 2, C, H, W]
        #neg_predictions1 = self.vggt.forward(negative1, query_points)
        #neg_conf1 = neg_predictions1["conf"]  #[B, 2, N]
        #neg_conf1 = torch.mean(neg_conf1[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative2 = torch.cat([query_image, negative_image2], dim=1)  #[B, 2, C, H, W]
        #neg_predictions2 = self.vggt.forward(negative2, query_points)
        #neg_conf2 = neg_predictions2["conf"]  #[B, 2, N]
        #neg_conf2 = torch.mean(neg_conf2[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        #batch_conf = torch.cat([pos_conf, neg_conf0, neg_conf1, neg_conf2], dim=-1)  #[B, 4]

        all_samples = torch.cat([positive, negative0, negative1, negative2], dim=0)  #[4*B, 2, C, H, W]
        all_query_points = query_points.repeat(4, 1, 1)  #[4*B, C, H, W]
        all_preds = self.vggt(all_samples, all_query_points)
        all_conf = all_preds["conf"]  #[4*B, 2, N]
        all_scores = torch.mean(all_conf[:, 1, :], dim=-1, keepdim=True)   #[4*B, 1]
        batch_conf = all_scores.view(4, B).permute(1, 0)  #[B, 4]

        labels = torch.zeros(B, dtype=torch.long).cuda()

        loss = self.criterion(batch_conf, labels)

        return loss
    
class VGGTforRerank3t1(nn.Module):
    def __init__(self, **kwargs):
        super(VGGTforRerank3t1, self).__init__(**kwargs)
        self.vggt = VGGT_adapter()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_points0, query_points1, query_points2, query_image0, query_image1, query_image2, positive_image, negative_image0, negative_image1, negative_image2):
        B = query_image0.size(0)
        positive0 = torch.cat([query_image0, positive_image], dim=1)  #[B, 2, C, H, W]
        #pos_predictions = self.vggt.forward(positive, query_points) 
        #pos_conf = pos_predictions["conf"]  #[B, 2, N]
        #pos_conf = torch.mean(pos_conf[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative00 = torch.cat([query_image0, negative_image0], dim=1)  #[B, 2, C, H, W]
        #neg_predictions0 = self.vggt.forward(negative0, query_points)
        #neg_conf0 = neg_predictions0["conf"]  #[B, 2, N]
        #neg_conf0 = torch.mean(neg_conf0[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative01 = torch.cat([query_image0, negative_image1], dim=1)  #[B, 2, C, H, W]
        #neg_predictions1 = self.vggt.forward(negative1, query_points)
        #neg_conf1 = neg_predictions1["conf"]  #[B, 2, N]
        #neg_conf1 = torch.mean(neg_conf1[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative02 = torch.cat([query_image0, negative_image2], dim=1)  #[B, 2, C, H, W]
        #neg_predictions2 = self.vggt.forward(negative2, query_points)
        #neg_conf2 = neg_predictions2["conf"]  #[B, 2, N]
        #neg_conf2 = torch.mean(neg_conf2[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        #batch_conf = torch.cat([pos_conf, neg_conf0, neg_conf1, neg_conf2], dim=-1)  #[B, 4]

        all_samples0 = torch.cat([positive0, negative00, negative01, negative02], dim=0)  #[4*B, 2, C, H, W]
        all_query_points0 = query_points0.repeat(4, 1, 1)  #[4*B, N, 2]
        all_preds0 = self.vggt(all_samples0, all_query_points0)
        all_conf0 = all_preds0["conf"]  #[4*B, 2, N]
        all_scores0 = torch.mean(all_conf0[:, 1, :], dim=-1, keepdim=True)   #[4*B, 1]
        batch_conf0 = all_scores0.view(4, B).permute(1, 0)  #[B, 4]

        positive1 = torch.cat([query_image1, positive_image], dim=1)  #[B, 2, C, H, W]
        #pos_predictions = self.vggt.forward(positive, query_points) 
        #pos_conf = pos_predictions["conf"]  #[B, 2, N]
        #pos_conf = torch.mean(pos_conf[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative10 = torch.cat([query_image1, negative_image0], dim=1)  #[B, 2, C, H, W]
        #neg_predictions0 = self.vggt.forward(negative0, query_points)
        #neg_conf0 = neg_predictions0["conf"]  #[B, 2, N]
        #neg_conf0 = torch.mean(neg_conf0[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative11 = torch.cat([query_image1, negative_image1], dim=1)  #[B, 2, C, H, W]
        #neg_predictions1 = self.vggt.forward(negative1, query_points)
        #neg_conf1 = neg_predictions1["conf"]  #[B, 2, N]
        #neg_conf1 = torch.mean(neg_conf1[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative12 = torch.cat([query_image1, negative_image2], dim=1)  #[B, 2, C, H, W]
        #neg_predictions2 = self.vggt.forward(negative2, query_points)
        #neg_conf2 = neg_predictions2["conf"]  #[B, 2, N]
        #neg_conf2 = torch.mean(neg_conf2[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        #batch_conf = torch.cat([pos_conf, neg_conf0, neg_conf1, neg_conf2], dim=-1)  #[B, 4]

        all_samples1 = torch.cat([positive1, negative10, negative11, negative12], dim=0)  #[4*B, 2, C, H, W]
        all_query_points1 = query_points1.repeat(4, 1, 1)  #[4*B, N, 2]
        all_preds1 = self.vggt(all_samples1, all_query_points1)
        all_conf1 = all_preds1["conf"]  #[4*B, 2, N]
        all_scores1 = torch.mean(all_conf1[:, 1, :], dim=-1, keepdim=True)   #[4*B, 1]
        batch_conf1 = all_scores1.view(4, B).permute(1, 0)  #[B, 4]

        positive2 = torch.cat([query_image2, positive_image], dim=1)  #[B, 2, C, H, W]
        #pos_predictions = self.vggt.forward(positive, query_points) 
        #pos_conf = pos_predictions["conf"]  #[B, 2, N]
        #pos_conf = torch.mean(pos_conf[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative20 = torch.cat([query_image2, negative_image0], dim=1)  #[B, 2, C, H, W]
        #neg_predictions0 = self.vggt.forward(negative0, query_points)
        #neg_conf0 = neg_predictions0["conf"]  #[B, 2, N]
        #neg_conf0 = torch.mean(neg_conf0[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative21 = torch.cat([query_image2, negative_image1], dim=1)  #[B, 2, C, H, W]
        #neg_predictions1 = self.vggt.forward(negative1, query_points)
        #neg_conf1 = neg_predictions1["conf"]  #[B, 2, N]
        #neg_conf1 = torch.mean(neg_conf1[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        negative22 = torch.cat([query_image2, negative_image2], dim=1)  #[B, 2, C, H, W]
        #neg_predictions2 = self.vggt.forward(negative2, query_points)
        #neg_conf2 = neg_predictions2["conf"]  #[B, 2, N]
        #neg_conf2 = torch.mean(neg_conf2[:, 1, :], dim=-1, keepdim=True)  #[B, 1]

        #batch_conf = torch.cat([pos_conf, neg_conf0, neg_conf1, neg_conf2], dim=-1)  #[B, 4]

        all_samples2 = torch.cat([positive2, negative20, negative21, negative22], dim=0)  #[4*B, 2, C, H, W]
        all_query_points2 = query_points2.repeat(4, 1, 1)  #[4*B, N, 2]
        all_preds2 = self.vggt(all_samples2, all_query_points2)
        all_conf2 = all_preds2["conf"]  #[4*B, 2, N]
        all_scores2 = torch.mean(all_conf2[:, 1, :], dim=-1, keepdim=True)   #[4*B, 1]
        batch_conf2 = all_scores2.view(4, B).permute(1, 0)  #[B, 4]

        batch_conf = (batch_conf0 + batch_conf1 + batch_conf2) / 3.0  #[B, 4]

        labels = torch.zeros(B, dtype=torch.long).to("cuda")

        loss = self.criterion(batch_conf, labels)

        return loss

class RerankClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super(RerankClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        self.norm = nn.LayerNorm(input_dim)
        self.weight = torch.tensor([1, 4], dtype=torch.float32, device='cuda')
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

        self.norm.apply(self._init_weights)
        self.classifier.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.mul_(init_scale)
            self.classifier.bias.data.mul_(init_scale)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features=None, labels=None, only_infer=False):
        logits = self.classifier(self.norm(features))  #[B, 2]

        if only_infer:
            return logits
        else:
            #labels_idx = torch.argmax(labels, dim=-1)
            loss = self.criterion(logits, labels)
            return loss
    
@register_model
def vggt_classifier(pretrained=False, **kwargs):
    model = RerankClassifier(**kwargs)
    return model
    
@register_model
def vggt_rerank(pretrained=False, **kwargs):
    model = VGGTforRerank(**kwargs)
    return model

@register_model
def vggt_rerank3t1(pretrained=False, **kwargs):
    model = VGGTforRerank3t1(**kwargs)
    return model