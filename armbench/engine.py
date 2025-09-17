import math
import os.path
import sys
import json
from typing import Iterable, Optional

from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from beit3_tools.beit3_datasets import get_sentencepiece_model_for_beit3
import numpy as np
from beit3_tools import utils
from tqdm import tqdm
import os
from vggt.utils.load_fn import load_and_preprocess_images
from sift import get_query_points
from vggt.models.vggt import VGGT
from vggt_adapter.models.vggt import VGGT_adapter
from rerank_modeling import RerankClassifier

class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class ArmbenchHandlerAllref(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None


    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        for i in range(len(labels)):
            if labels[i] in [ref_item_ids[index] for index in topk[i].tolist()]:
                correct += 1.0

        return correct / len(labels)

    def calculate_accuracy_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = Counter([ref_item_ids[index] for index in topk[i].tolist()]).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        # topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk3 = scores.topk(3, dim=1)
        topk2 = scores.topk(2, dim=1)
        topk1 = scores.topk(1, dim=1)

        # topk10_iids = refids[topk10.indices]
        topk5_iids = refids[topk5.indices]
        topk3_iids = refids[topk3.indices]
        topk2_iids = refids[topk2.indices]
        topk1_iids = refids[topk1.indices]

        # topk10_iids = topk10_iids.detach().cpu()
        topk5_iids = topk5_iids.detach().cpu()
        topk3_iids = topk3_iids.detach().cpu()
        topk2_iids = topk2_iids.detach().cpu()
        topk1_iids = topk1_iids.detach().cpu()

        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(topk5_iids, labels, ref_item_ids, k=5)
        tr_r3 = self.calculate_recall_at_k(topk3_iids, labels, ref_item_ids, k=3)
        tr_r2 = self.calculate_recall_at_k(topk2_iids, labels, ref_item_ids, k=2)
        tr_r1 = self.calculate_recall_at_k(topk1_iids, labels, ref_item_ids, k=1)

        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(topk5_iids, labels, ref_item_ids, k=5)
        acc_r3 = self.calculate_accuracy_at_k(topk3_iids, labels, ref_item_ids, k=3)
        acc_r2 = self.calculate_accuracy_at_k(topk2_iids, labels, ref_item_ids, k=2)
        acc_r1 = self.calculate_accuracy_at_k(topk1_iids, labels, ref_item_ids, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())


class ArmbenchHandlerAllrefVGGT(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.correct_recall_dict = {}
        self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.classifier = RerankClassifier()
        self.vggt.to("cuda").eval()
        self.classifier.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.correct_recall_dict.clear()
        self.correct_accuracy_dict.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k_per_pick(self, topk, label, ref_item_ids, k):
        if label in [ref_item_ids[index] for index in topk.tolist()]:
            self.correct_recall_dict[f"{k}"] += 1.0

    def calculate_accuracy_at_k_per_pick(self, topk, label, ref_item_ids, k):
        from collections import Counter
        predict_label = Counter([ref_item_ids[index] for index in topk.tolist()]).most_common(1)[0][0]
        if predict_label == label:
            self.correct_accuracy_dict[f"{k}"] += 1.0

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        self.correct_recall_dict["1"] = 0.0
        self.correct_recall_dict["2"] = 0.0
        self.correct_recall_dict["3"] = 0.0
        self.correct_recall_dict["5"] = 0.0
        self.correct_accuracy_dict["1"] = 0.0
        self.correct_accuracy_dict["2"] = 0.0
        self.correct_accuracy_dict["3"] = 0.0
        self.correct_accuracy_dict["5"] = 0.0
        topk16 = scores.topk(16, dim=1)  #[len(pick_ids), 16]
        topk16_iids = refids[topk16.indices].to("cuda")
        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size
        for i in tqdm(range(len(pickids)), desc="VGGT reranking"):
            label = labels[i]
            query_feat = query_cls_feats[i].unsqueeze(0).to('cuda')
            logits = self.classifier(features=query_feat, only_infer=True)
            predict = torch.argmax(logits, dim=1)
            if predict.item() == 0:
                topk5_iids = topk16_iids[i][:5]
                topk3_iids = topk16_iids[i][:3]
                topk2_iids = topk16_iids[i][:2]
                topk1_iids = topk16_iids[i][:1]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()
            else:
                pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
                query_image_path = query_dataloader.dataset._get_image_path(pick_id)
                query_points = get_query_points(query_image_path)  #[N, 2]  (N=20)
                conf_per_pick = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                    batch_image_list = []
                    #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                    #ref_image_paths = []
                    #for k in range(len(ref_ids)):
                    #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                    #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                    for index in topk16_iids[i][start_idx:end_idx].tolist():
                        image_path = [query_image_path]
                        ref_image_path = ref_dataloader.dataset._get_image_path(index)
                        image_path.extend([ref_image_path])
                        images = load_and_preprocess_images(image_path, mode="pad")  #[2, C, H, W]
                        batch_image_list.append(images)
                    batch_images = torch.stack(batch_image_list, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_query_points = query_points.repeat(batch_images.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    with torch.amp.autocast("cuda"):
                        predictions = self.vggt.forward(batch_images, batch_query_points)
                    batch_conf = predictions["conf"]  #[B, 2, N]
                    conf = torch.mean(batch_conf[:, 1, :], dim=-1, keepdim=False)  #[B]
                    conf_per_pick.append(conf)
                conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
                #conf_per_pick = self.threshold_process(conf_per_pick)

                topk5 = conf_per_pick.topk(5, dim=0)
                topk3 = conf_per_pick.topk(3, dim=0)
                topk2 = conf_per_pick.topk(2, dim=0)
                topk1 = conf_per_pick.topk(1, dim=0)

                topk5_iids = topk16_iids[i][topk5.indices]
                topk3_iids = topk16_iids[i][topk3.indices]
                topk2_iids = topk16_iids[i][topk2.indices]
                topk1_iids = topk16_iids[i][topk1.indices]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()

            self.calculate_recall_at_k_per_pick(topk5_iids, label, ref_item_ids, k=5)
            self.calculate_recall_at_k_per_pick(topk3_iids, label, ref_item_ids, k=3)
            self.calculate_recall_at_k_per_pick(topk2_iids, label, ref_item_ids, k=2)
            self.calculate_recall_at_k_per_pick(topk1_iids, label, ref_item_ids, k=1)

            self.calculate_accuracy_at_k_per_pick(topk5_iids, label, ref_item_ids, k=5)
            self.calculate_accuracy_at_k_per_pick(topk3_iids, label, ref_item_ids, k=3)
            self.calculate_accuracy_at_k_per_pick(topk2_iids, label, ref_item_ids, k=2)
            self.calculate_accuracy_at_k_per_pick(topk1_iids, label, ref_item_ids, k=1)

        tr_r1 = self.correct_recall_dict["1"] / len(labels)
        tr_r2 = self.correct_recall_dict["2"] / len(labels)
        tr_r3 = self.correct_recall_dict["3"] / len(labels)
        tr_r5 = self.correct_recall_dict["5"] / len(labels)

        acc_r1 = self.correct_accuracy_dict["1"] / len(labels)
        acc_r2 = self.correct_accuracy_dict["2"] / len(labels)
        acc_r3 = self.correct_accuracy_dict["3"] / len(labels)
        acc_r5 = self.correct_accuracy_dict["5"] / len(labels)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandlerAllrefVGGTextract(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.vggt = VGGT()
        self.vggt.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def save_sample_paths(self, sample_paths, filename):
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sample_paths, f, ensure_ascii=False, indent=4)
        print(f"* save paths to {filename}")

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        sample_list = []
        topk16 = scores.topk(16, dim=1) #[len(pick_ids), 16]
        topk16_iids = refids[topk16.indices].to("cuda")
        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size
        for i in tqdm(range(len(pickids)), desc="VGGT reranking"):
            pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
            query_image_path = query_dataloader.dataset._get_image_path(pick_id)
            query_points = get_query_points(query_image_path)  #[N, 2]  (N=20)
            conf_per_pick = []
            label = labels[i]
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                batch_image_list = []
                #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                #ref_image_paths = []
                #for k in range(len(ref_ids)):
                #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                for index in topk16_iids[i][start_idx:end_idx].tolist():
                    image_path = [query_image_path]
                    ref_image_path = ref_dataloader.dataset._get_image_path(index)
                    image_path.extend([ref_image_path])
                    images = load_and_preprocess_images(image_path, mode="pad")  #[2, C, H, W]
                    batch_image_list.append(images)
                batch_images = torch.stack(batch_image_list, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_query_points = query_points.repeat(batch_images.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                predictions = self.vggt.forward(batch_images, batch_query_points)
                batch_conf = predictions["conf"]  #[B, 2, N]
                conf = torch.mean(batch_conf[:, 1, :], dim=-1, keepdim=False)  #[B]
                conf_per_pick.append(conf)
            conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]

            same_id_mask = torch.tensor(
            [ref_item_ids[idx] == label for idx in topk16_iids[i].tolist()],
            device=conf_per_pick.device)

            if torch.any(same_id_mask):
                pos_conf, pos_idx = conf_per_pick[same_id_mask].max(dim=0)
                positive_idx = topk16_iids[i][same_id_mask.nonzero()[pos_idx].item()]
                positive_path = ref_dataloader.dataset._get_image_path(positive_idx)
            else:
                paths = ref_dataloader.dataset.items[label]
                if len(paths) > 1:
                    path = random.sample(paths, 1)
                    positive_path = os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', label, path[0])
                else:
                    positive_path = os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', label, paths[0])

            negative_mask = torch.logical_not(same_id_mask)
            if torch.any(negative_mask):
                neg_confs = conf_per_pick[negative_mask]
                neg_indices = torch.tensor(topk16_iids[i])[negative_mask]
                _, top3_idx = torch.topk(neg_confs, k=3)
                negative_paths = [
                    ref_dataloader.dataset._get_image_path(neg_indices[i].item())
                    for i in top3_idx.tolist()
                ]
            else:
                negative_paths = None

            sample_list.append({
            "query_path": query_image_path.replace("\\", "/"),   #string
            "positive_path": positive_path.replace("\\", "/"),   #string
            "negative_paths": [path.replace("\\", "/") for path in negative_paths]   #list
            })

        self.save_sample_paths(sample_list, "sample_paths_full.json")
        exit(0)

        return None, None

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandlerAllrefVGGTsample(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        #self.correct_recall_dict = {}
        self.weights = {1:0.5, 2:0.3, 3:0.2}
        self.delta = 0.01
        self.feature_list = []
        self.label_list = []
        #self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.vggt.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        #self.correct_recall_dict.clear()
        #self.correct_accuracy_dict.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_metrics(self, topk_ids, label, ref_item_ids, metrics_dict, k):
        """calculate Recall@k per pick"""
        # Recall@k
        if label in [ref_item_ids[i] for i in topk_ids.tolist()]:
            metrics_dict["recall"][k] += 1
        
        # Accuracy@k
        #from collections import Counter
        #pred_label = Counter([ref_item_ids[i] for i in topk_ids.tolist()]).most_common(1)[0][0]
        #if pred_label == label:
        #    metrics_dict["accuracy"][k] += 1

    def _calculate_top_score(self, metrics_dict):
        """calculate top score (0.5*R@1 + 0.3*R@2 + 0.2*R@3)"""
        return sum(self.weights[k] * metrics_dict["recall"][k] for k in self.weights.keys())

    #def threshold_process(self, input, threshold=0.03):
    #    if torch.all(input < threshold):
    #        sorted_input, _ = torch.sort(input, descending=True)
    #        return sorted_input
    #    else:
    #        return input
        
    def _save_samples(self, data_path):
        features_np = np.array(self.feature_list)  # [N, 768]
        print(f"features shape: {features_np.shape}")
        labels_np = np.array(self.label_list)      # [N, 1]
        print(f"labels shape: {labels_np.shape}")
        
        np.save(os.path.join(data_path, "features.npy").replace('\\', '/'), features_np)
        np.save(os.path.join(data_path, "labels.npy").replace('\\', '/'), labels_np)
        
        print(f"save: {len(features_np)} features, {len(labels_np)} labels")

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        topk16 = scores.topk(16, dim=1)
        topk16_iids = refids[topk16.indices].to("cuda")

        positive_samples = 0
        negative_samples = 0
        metrics_2d = {"recall": {k: 0.0 for k in [1, 2, 3]}}
        metrics_rerank = {"recall": {k: 0.0 for k in [1, 2, 3]}}
        metrics_2d_test = {"recall": {k: 0.0 for k in [1, 2, 3]}}
        metrics_rerank_test = {"recall": {k: 0.0 for k in [1, 2, 3]}}

        for i in tqdm(range(len(pickids)), desc="extracting"):

            for k in [1, 2, 3]:
                metrics_2d["recall"][k] = 0.0
                metrics_rerank["recall"][k] = 0.0

            label = labels[i]
            topk3_2d = topk16_iids[i][:3]
            topk2_2d = topk16_iids[i][:2]
            topk1_2d = topk16_iids[i][:1]

            topk3_2d = topk3_2d.detach().cpu()
            topk2_2d = topk2_2d.detach().cpu()
            topk1_2d = topk1_2d.detach().cpu()

            self.calculate_metrics(topk3_2d, label, ref_item_ids, metrics_2d, k=3)
            self.calculate_metrics(topk2_2d, label, ref_item_ids, metrics_2d, k=2)
            self.calculate_metrics(topk1_2d, label, ref_item_ids, metrics_2d, k=1)
            original_score = self._calculate_top_score(metrics_2d)
            self.calculate_metrics(topk3_2d, label, ref_item_ids, metrics_2d_test, k=3)
            self.calculate_metrics(topk2_2d, label, ref_item_ids, metrics_2d_test, k=2)
            self.calculate_metrics(topk1_2d, label, ref_item_ids, metrics_2d_test, k=1)

            candidate_len = 16
            batch_size = 16
            total_batches = (candidate_len + batch_size - 1) // batch_size

            pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
            query_image_path = query_dataloader.dataset._get_image_path(pick_id)
            query_points = get_query_points(query_image_path)  #[N, 2]  (N=20)
            conf_per_pick = []
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                batch_image_list = []
                #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                #ref_image_paths = []
                #for k in range(len(ref_ids)):
                #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                for index in topk16_iids[i][start_idx:end_idx].tolist():
                    image_path = [query_image_path]
                    ref_image_path = ref_dataloader.dataset._get_image_path(index)
                    image_path.extend([ref_image_path])
                    images = load_and_preprocess_images(image_path, mode="pad")  #[2, C, H, W]
                    batch_image_list.append(images)
                batch_images = torch.stack(batch_image_list, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_query_points = query_points.repeat(batch_images.size(0), 1, 1).float().to("cuda")   #[B, N ,2]
                predictions = self.vggt.forward(batch_images, batch_query_points)
                batch_conf = predictions["conf"]  #[B, 2, N]
                #batch_vis = predictions["vis"]  #[B, 2, N]
                #batch_combined = 0.1*batch_vis + 0.9*batch_conf  #[B, 2, N]
                #print(f"batch_conf:{batch_conf[:, 1, :]}")
                #batch_conf_topk = torch.topk(batch_conf[:, 1, :], 18, dim=-1)
                #conf = torch.mean(batch_conf_topk.values, dim=-1, keepdim=False)  #[B]
                conf = torch.mean(batch_conf[:, 1, :], dim=-1, keepdim=False)  #[B]
                #conf = torch.mean(batch_combined[:, 1, :], dim=-1, keepdim=False)  #[B]
                conf_per_pick.append(conf)
            conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
            #conf_per_pick = self.threshold_process(conf_per_pick)
            #print(f"conf_per_pick:{conf_per_pick}")

            topk3 = conf_per_pick.topk(3, dim=0)
            topk2 = conf_per_pick.topk(2, dim=0)
            topk1 = conf_per_pick.topk(1, dim=0)

            topk3_iids = topk16_iids[i][topk3.indices]
            topk2_iids = topk16_iids[i][topk2.indices]
            topk1_iids = topk16_iids[i][topk1.indices]

            topk3_iids = topk3_iids.detach().cpu()
            topk2_iids = topk2_iids.detach().cpu()
            topk1_iids = topk1_iids.detach().cpu()

            self.calculate_metrics(topk3_iids, label, ref_item_ids, metrics_rerank, k=3)
            self.calculate_metrics(topk2_iids, label, ref_item_ids, metrics_rerank, k=2)
            self.calculate_metrics(topk1_iids, label, ref_item_ids, metrics_rerank, k=1)
            rerank_score = self._calculate_top_score(metrics_rerank)
            self.calculate_metrics(topk3_iids, label, ref_item_ids, metrics_rerank_test, k=3)
            self.calculate_metrics(topk2_iids, label, ref_item_ids, metrics_rerank_test, k=2)
            self.calculate_metrics(topk1_iids, label, ref_item_ids, metrics_rerank_test, k=1)

            score_diff = rerank_score - original_score
            label_rerank = 1 if score_diff > self.delta else 0
            self.label_list.append(label_rerank)

            features = query_cls_feats[i]

            self.feature_list.append(features.cpu().numpy())

            if label_rerank == 1:
                positive_samples += 1
                #self.label_list.append([1, 0])
            else:
                negative_samples += 1
                #self.label_list.append([0, 1])

        self._save_samples(ref_dataloader.dataset.data_path)
        print(f"  positive samples: {positive_samples}")
        print(f"  negative samples: {negative_samples}")

        rerank_tr_r1 = metrics_rerank_test["recall"][1] / len(labels)
        rerank_tr_r2 = metrics_rerank_test["recall"][2] / len(labels)
        rerank_tr_r3 = metrics_rerank_test["recall"][3] / len(labels)

        base_tr_r1 = metrics_2d_test["recall"][1] / len(labels)
        base_tr_r2 = metrics_2d_test["recall"][2] / len(labels)
        base_tr_r3 = metrics_2d_test["recall"][3] / len(labels)

        eval_result = {
            "rerank_tr_r1": rerank_tr_r1 * 100.0,
            "rerank_tr_r2": rerank_tr_r2 * 100.0,
            "rerank_tr_r3": rerank_tr_r3 * 100.0,

            "rerank_average_score": (rerank_tr_r1 + rerank_tr_r2 + rerank_tr_r3) / 3.0,

            "base_tr_r1": base_tr_r1 * 100.0,
            "base_tr_r2": base_tr_r2 * 100.0,
            "base_tr_r3": base_tr_r3 * 100.0,

            "base_average_score": (base_tr_r1 + base_tr_r2 + base_tr_r3) / 3.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "rerank_average_score"
        

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandlerAllrefVGGT3t1(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.correct_recall_dict = {}
        self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.classifier = RerankClassifier()
        self.vggt.to("cuda").eval()
        self.classifier.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.correct_recall_dict.clear()
        self.correct_accuracy_dict.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k_per_pick(self, topk, label, ref_item_ids, k):
        if label in [ref_item_ids[index] for index in topk.tolist()]:
            self.correct_recall_dict[f"{k}"] += 1.0

    def calculate_accuracy_at_k_per_pick(self, topk, label, ref_item_ids, k):
        from collections import Counter
        predict_label = Counter([ref_item_ids[index] for index in topk.tolist()]).most_common(1)[0][0]
        if predict_label == label:
            self.correct_accuracy_dict[f"{k}"] += 1.0

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        self.correct_recall_dict["1"] = 0.0
        self.correct_recall_dict["2"] = 0.0
        self.correct_recall_dict["3"] = 0.0
        self.correct_recall_dict["5"] = 0.0
        self.correct_accuracy_dict["1"] = 0.0
        self.correct_accuracy_dict["2"] = 0.0
        self.correct_accuracy_dict["3"] = 0.0
        self.correct_accuracy_dict["5"] = 0.0
        topk16 = scores.topk(16, dim=1)  #[len(pick_ids), 16]
        topk16_iids = refids[topk16.indices].to("cuda")
        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size
        for i in tqdm(range(len(pickids)), desc="VGGT reranking"):
            label = labels[i]
            query_feat = query_cls_feats[i].unsqueeze(0).to('cuda')
            logits = self.classifier(features=query_feat, only_infer=True)
            predict = torch.argmax(logits, dim=1)
            if predict.item() == 0:
                topk5_iids = topk16_iids[i][:5]
                topk3_iids = topk16_iids[i][:3]
                topk2_iids = topk16_iids[i][:2]
                topk1_iids = topk16_iids[i][:1]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()
            else:
                pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
                query_image_path = query_dataloader.dataset._get_image_path(pick_id)
                query_points0 = get_query_points(query_image_path[0])  #[N, 2]  (N=20)
                query_points1 = get_query_points(query_image_path[1])  #[N, 2]  (N=20)
                query_points2 = get_query_points(query_image_path[2])  #[N, 2]  (N=20)
                conf_per_pick = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                    batch_image_list0 = []
                    batch_image_list1 = []
                    batch_image_list2 = []
                    #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                    #ref_image_paths = []
                    #for k in range(len(ref_ids)):
                    #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                    #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                    for index in topk16_iids[i][start_idx:end_idx].tolist():
                        image_path0 = [query_image_path[0]]
                        image_path1 = [query_image_path[1]]
                        image_path2 = [query_image_path[2]]
                        ref_image_path = ref_dataloader.dataset._get_image_path(index)
                        image_path0.extend([ref_image_path])
                        image_path1.extend([ref_image_path])
                        image_path2.extend([ref_image_path])
                        images0 = load_and_preprocess_images(image_path0, mode="pad")  #[2, C, H, W]
                        images1 = load_and_preprocess_images(image_path1, mode="pad")  #[2, C, H, W]
                        images2 = load_and_preprocess_images(image_path2, mode="pad")  #[2, C, H, W]
                        batch_image_list0.append(images0)
                        batch_image_list1.append(images1)
                        batch_image_list2.append(images2)
                    batch_images0 = torch.stack(batch_image_list0, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_images1 = torch.stack(batch_image_list1, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_images2 = torch.stack(batch_image_list2, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_query_points0 = query_points0.repeat(batch_images0.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    batch_query_points1 = query_points1.repeat(batch_images1.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    batch_query_points2 = query_points2.repeat(batch_images2.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    predictions0 = self.vggt.forward(batch_images0, batch_query_points0)
                    batch_conf0 = predictions0["conf"]  #[B, 2, N]
                    conf0 = torch.mean(batch_conf0[:, 1, :], dim=-1, keepdim=False)  #[B]

                    predictions1 = self.vggt.forward(batch_images1, batch_query_points1)
                    batch_conf1 = predictions1["conf"]  #[B, 2, N]
                    conf1 = torch.mean(batch_conf1[:, 1, :], dim=-1, keepdim=False)  #[B]

                    predictions2 = self.vggt.forward(batch_images2, batch_query_points2)
                    batch_conf2 = predictions2["conf"]  #[B, 2, N]
                    conf2 = torch.mean(batch_conf2[:, 1, :], dim=-1, keepdim=False)  #[B]

                    conf = (conf0 + conf1 + conf2) / 3.0
                    conf_per_pick.append(conf)
                conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
                #conf_per_pick = self.threshold_process(conf_per_pick)

                topk5 = conf_per_pick.topk(5, dim=0)
                topk3 = conf_per_pick.topk(3, dim=0)
                topk2 = conf_per_pick.topk(2, dim=0)
                topk1 = conf_per_pick.topk(1, dim=0)

                topk5_iids = topk16_iids[i][topk5.indices]
                topk3_iids = topk16_iids[i][topk3.indices]
                topk2_iids = topk16_iids[i][topk2.indices]
                topk1_iids = topk16_iids[i][topk1.indices]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()

            self.calculate_recall_at_k_per_pick(topk5_iids, label, ref_item_ids, k=5)
            self.calculate_recall_at_k_per_pick(topk3_iids, label, ref_item_ids, k=3)
            self.calculate_recall_at_k_per_pick(topk2_iids, label, ref_item_ids, k=2)
            self.calculate_recall_at_k_per_pick(topk1_iids, label, ref_item_ids, k=1)

            self.calculate_accuracy_at_k_per_pick(topk5_iids, label, ref_item_ids, k=5)
            self.calculate_accuracy_at_k_per_pick(topk3_iids, label, ref_item_ids, k=3)
            self.calculate_accuracy_at_k_per_pick(topk2_iids, label, ref_item_ids, k=2)
            self.calculate_accuracy_at_k_per_pick(topk1_iids, label, ref_item_ids, k=1)

        tr_r1 = self.correct_recall_dict["1"] / len(labels)
        tr_r2 = self.correct_recall_dict["2"] / len(labels)
        tr_r3 = self.correct_recall_dict["3"] / len(labels)
        tr_r5 = self.correct_recall_dict["5"] / len(labels)

        acc_r1 = self.correct_accuracy_dict["1"] / len(labels)
        acc_r2 = self.correct_accuracy_dict["2"] / len(labels)
        acc_r3 = self.correct_accuracy_dict["3"] / len(labels)
        acc_r5 = self.correct_accuracy_dict["5"] / len(labels)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())


class ArmbenchHandlerAllrefVGGT3t1extract(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.vggt = VGGT()
        self.vggt.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_metrics(self, topk_ids, label, ref_item_ids, metrics_dict, k):
        """calculate Recall@k and Accuracy@k per pick"""
        # Recall@k
        if label in [ref_item_ids[i] for i in topk_ids.tolist()]:
            metrics_dict["recall"][k] += 1
        
        # Accuracy@k
        from collections import Counter
        pred_label = Counter([ref_item_ids[i] for i in topk_ids.tolist()]).most_common(1)[0][0]
        if pred_label == label:
            metrics_dict["accuracy"][k] += 1

    def save_sample_paths(self, sample_paths, filename):
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sample_paths, f, ensure_ascii=False, indent=4)
        print(f"* save paths to {filename}")

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        metrics_2d = {"recall": {k: 0.0 for k in [1, 2, 3, 5]}, "accuracy": {k: 0.0 for k in [1, 2, 3, 5]}}
        metrics_rerank = {"recall": {k: 0.0 for k in [1, 2, 3, 5]}, "accuracy": {k: 0.0 for k in [1, 2, 3, 5]}}
        sample_list = []
        topk16 = scores.topk(16, dim=1) #[len(pick_ids), 16]
        topk16_iids = refids[topk16.indices].to("cuda")
        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size
        for i in tqdm(range(len(pickids)), desc="VGGT reranking"):
            label = labels[i]
            topk5_2d = topk16_iids[i][:5]
            topk3_2d = topk16_iids[i][:3]
            topk2_2d = topk16_iids[i][:2]
            topk1_2d = topk16_iids[i][:1]

            topk5_2d = topk5_2d.detach().cpu()
            topk3_2d = topk3_2d.detach().cpu()
            topk2_2d = topk2_2d.detach().cpu()
            topk1_2d = topk1_2d.detach().cpu()

            self.calculate_metrics(topk5_2d, label, ref_item_ids, metrics_2d, k=5)
            self.calculate_metrics(topk3_2d, label, ref_item_ids, metrics_2d, k=3)
            self.calculate_metrics(topk2_2d, label, ref_item_ids, metrics_2d, k=2)
            self.calculate_metrics(topk1_2d, label, ref_item_ids, metrics_2d, k=1)

            pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
            query_image_path = query_dataloader.dataset._get_image_path(pick_id)
            query_points0 = get_query_points(query_image_path[0])  #[N, 2]  (N=20)
            query_points1 = get_query_points(query_image_path[1])  #[N, 2]  (N=20)
            query_points2 = get_query_points(query_image_path[2])  #[N, 2]  (N=20)
            conf_per_pick = []
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                batch_image_list0 = []
                batch_image_list1 = []
                batch_image_list2 = []
                #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                #ref_image_paths = []
                #for k in range(len(ref_ids)):
                #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                for index in topk16_iids[i][start_idx:end_idx].tolist():
                    image_path0 = [query_image_path[0]]
                    image_path1 = [query_image_path[1]]
                    image_path2 = [query_image_path[2]]
                    ref_image_path = ref_dataloader.dataset._get_image_path(index)
                    image_path0.extend([ref_image_path])
                    image_path1.extend([ref_image_path])
                    image_path2.extend([ref_image_path])
                    images0 = load_and_preprocess_images(image_path0, mode="pad")  #[2, C, H, W]
                    images1 = load_and_preprocess_images(image_path1, mode="pad")  #[2, C, H, W]
                    images2 = load_and_preprocess_images(image_path2, mode="pad")  #[2, C, H, W]
                    batch_image_list0.append(images0)
                    batch_image_list1.append(images1)
                    batch_image_list2.append(images2)
                batch_images0 = torch.stack(batch_image_list0, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_images1 = torch.stack(batch_image_list1, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_images2 = torch.stack(batch_image_list2, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_query_points0 = query_points0.repeat(batch_images0.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                batch_query_points1 = query_points1.repeat(batch_images1.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                batch_query_points2 = query_points2.repeat(batch_images2.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                predictions0 = self.vggt.forward(batch_images0, batch_query_points0)
                batch_conf0 = predictions0["conf"]  #[B, 2, N]
                conf0 = torch.mean(batch_conf0[:, 1, :], dim=-1, keepdim=False)  #[B]

                predictions1 = self.vggt.forward(batch_images1, batch_query_points1)
                batch_conf1 = predictions1["conf"]  #[B, 2, N]
                conf1 = torch.mean(batch_conf1[:, 1, :], dim=-1, keepdim=False)  #[B]

                predictions2 = self.vggt.forward(batch_images2, batch_query_points2)
                batch_conf2 = predictions2["conf"]  #[B, 2, N]
                conf2 = torch.mean(batch_conf2[:, 1, :], dim=-1, keepdim=False)  #[B]

                conf = (conf0 + conf1 + conf2) / 3.0
                conf_per_pick.append(conf)
            conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]

            topk5 = conf_per_pick.topk(5, dim=0)
            topk3 = conf_per_pick.topk(3, dim=0)
            topk2 = conf_per_pick.topk(2, dim=0)
            topk1 = conf_per_pick.topk(1, dim=0)

            topk5_iids = topk16_iids[i][topk5.indices]
            topk3_iids = topk16_iids[i][topk3.indices]
            topk2_iids = topk16_iids[i][topk2.indices]
            topk1_iids = topk16_iids[i][topk1.indices]

            topk5_iids = topk5_iids.detach().cpu()
            topk3_iids = topk3_iids.detach().cpu()
            topk2_iids = topk2_iids.detach().cpu()
            topk1_iids = topk1_iids.detach().cpu()

            self.calculate_metrics(topk5_iids, label, ref_item_ids, metrics_rerank, k=5)
            self.calculate_metrics(topk3_iids, label, ref_item_ids, metrics_rerank, k=3)
            self.calculate_metrics(topk2_iids, label, ref_item_ids, metrics_rerank, k=2)
            self.calculate_metrics(topk1_iids, label, ref_item_ids, metrics_rerank, k=1)

            same_id_mask = torch.tensor(
            [ref_item_ids[idx] == label for idx in topk16_iids[i].tolist()],
            device=conf_per_pick.device)

            if torch.any(same_id_mask):
                pos_conf, pos_idx = conf_per_pick[same_id_mask].max(dim=0)
                positive_idx = topk16_iids[i][same_id_mask.nonzero()[pos_idx].item()]
                positive_path = ref_dataloader.dataset._get_image_path(positive_idx)
            else:
                paths = ref_dataloader.dataset.items[label]
                if len(paths) > 1:
                    path = random.sample(paths, 1)
                    positive_path = os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', label, path[0])
                else:
                    positive_path = os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', label, paths[0])

            negative_mask = torch.logical_not(same_id_mask)
            if torch.any(negative_mask):
                neg_confs = conf_per_pick[negative_mask]
                neg_indices = torch.tensor(topk16_iids[i])[negative_mask]
                _, top3_idx = torch.topk(neg_confs, k=3)
                negative_paths = [
                    ref_dataloader.dataset._get_image_path(neg_indices[i].item())
                    for i in top3_idx.tolist()
                ]
            else:
                negative_paths = None

            sample_list.append({
            "query_path0": query_image_path[0].replace("\\", "/"),   #string
            "query_path1": query_image_path[1].replace("\\", "/"),   #string
            "query_path2": query_image_path[2].replace("\\", "/"),   #string
            "positive_path": positive_path.replace("\\", "/"),   #string
            "negative_paths": [path.replace("\\", "/") for path in negative_paths]   #list
            })

        self.save_sample_paths(sample_list, "sample_paths_full_3t1.json")

        rerank_tr_r1 = metrics_rerank["recall"][1] / len(labels)
        rerank_tr_r2 = metrics_rerank["recall"][2] / len(labels)
        rerank_tr_r3 = metrics_rerank["recall"][3] / len(labels)
        rerank_tr_r5 = metrics_rerank["recall"][5] / len(labels)

        rerank_acc_r1 = metrics_rerank["accuracy"][1] / len(labels)
        rerank_acc_r2 = metrics_rerank["accuracy"][2] / len(labels)
        rerank_acc_r3 = metrics_rerank["accuracy"][3] / len(labels)
        rerank_acc_r5 = metrics_rerank["accuracy"][5] / len(labels)

        base_tr_r1 = metrics_2d["recall"][1] / len(labels)
        base_tr_r2 = metrics_2d["recall"][2] / len(labels)
        base_tr_r3 = metrics_2d["recall"][3] / len(labels)
        base_tr_r5 = metrics_2d["recall"][5] / len(labels)

        base_acc_r1 = metrics_2d["accuracy"][1] / len(labels)
        base_acc_r2 = metrics_2d["accuracy"][2] / len(labels)
        base_acc_r3 = metrics_2d["accuracy"][3] / len(labels)
        base_acc_r5 = metrics_2d["accuracy"][5] / len(labels)

        eval_result = {
            "rerank_tr_r1": rerank_tr_r1 * 100.0,
            "rerank_tr_r2": rerank_tr_r2 * 100.0,
            "rerank_tr_r3": rerank_tr_r3 * 100.0,
            "rerank_tr_r5": rerank_tr_r5 * 100.0,


            "rerank_acc_r1": rerank_acc_r1 * 100.0,
            "rerank_acc_r2": rerank_acc_r2 * 100.0,
            "rerank_acc_r3": rerank_acc_r3 * 100.0,
            "rerank_acc_r5": rerank_acc_r5 * 100.0,

            "rerank_average_score": (rerank_tr_r1 + rerank_tr_r2 + rerank_tr_r3 + rerank_tr_r5 + rerank_acc_r1 + rerank_acc_r2 + rerank_acc_r3 + rerank_acc_r5) / 8.0,

            "base_tr_r1": base_tr_r1 * 100.0,
            "base_tr_r2": base_tr_r2 * 100.0,
            "base_tr_r3": base_tr_r3 * 100.0,
            "base_tr_r5": base_tr_r5 * 100.0,


            "base_acc_r1": base_acc_r1 * 100.0,
            "base_acc_r2": base_acc_r2 * 100.0,
            "base_acc_r3": base_acc_r3 * 100.0,
            "base_acc_r5": base_acc_r5 * 100.0,

            "base_average_score": (base_tr_r1 + base_tr_r2 + base_tr_r3 + base_tr_r5 + base_acc_r1 + base_acc_r2 + base_acc_r3 + base_acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "rerank_average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())


class ArmbenchHandlerAllrefVGGT3t1sample(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        #self.correct_recall_dict = {}
        self.weights = {1:0.5, 2:0.3, 3:0.2}
        self.delta = 0.01
        self.label_list = []
        #self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.vggt.to("cuda").eval()

    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        #self.correct_recall_dict.clear()
        #self.correct_accuracy_dict.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_metrics(self, topk_ids, label, ref_item_ids, metrics_dict, k):
        """calculate Recall@k per pick"""
        # Recall@k
        if label in [ref_item_ids[i] for i in topk_ids.tolist()]:
            metrics_dict["recall"][k] += 1
        
        # Accuracy@k
        #from collections import Counter
        #pred_label = Counter([ref_item_ids[i] for i in topk_ids.tolist()]).most_common(1)[0][0]
        #if pred_label == label:
        #    metrics_dict["accuracy"][k] += 1

    def _calculate_top_score(self, metrics_dict):
        """calculate top score (0.5*R@1 + 0.3*R@2 + 0.2*R@3)"""
        return sum(self.weights[k] * metrics_dict["recall"][k] for k in self.weights.keys())

    #def threshold_process(self, input, threshold=0.03):
    #    if torch.all(input < threshold):
    #        sorted_input, _ = torch.sort(input, descending=True)
    #        return sorted_input
    #    else:
    #        return input
        
    def _save_samples(self, data_path, query_cls_feat):
        features_np = np.array(query_cls_feat)  # [N, 768]
        print(f"features shape: {features_np.shape}")
        labels_np = np.array(self.label_list)      # [N, 1]
        print(f"labels shape: {labels_np.shape}")
        
        np.save(os.path.join(data_path, "features_3t1.npy").replace('\\', '/'), features_np)
        np.save(os.path.join(data_path, "labels_3t1.npy").replace('\\', '/'), labels_np)
        
        print(f"save: {len(features_np)} features, {len(labels_np)} labels")

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        topk16 = scores.topk(16, dim=1)
        topk16_iids = refids[topk16.indices].to("cuda")

        positive_samples = 0
        negative_samples = 0
        metrics_2d = {"recall": {k: 0.0 for k in [1, 2, 3]}}
        metrics_rerank = {"recall": {k: 0.0 for k in [1, 2, 3]}}

        metrics_2d_score = {"recall": {k: 0.0 for k in [1, 2, 3]}}
        metrics_rerank_score = {"recall": {k: 0.0 for k in [1, 2, 3]}}

        for i in tqdm(range(len(pickids)), desc="extracting"):
            for k in [1, 2, 3]:
                metrics_2d["recall"][k] = 0.0
                metrics_rerank["recall"][k] = 0.0

            label = labels[i]
            topk3_2d = topk16_iids[i][:3]
            topk2_2d = topk16_iids[i][:2]
            topk1_2d = topk16_iids[i][:1]

            topk3_2d = topk3_2d.detach().cpu()
            topk2_2d = topk2_2d.detach().cpu()
            topk1_2d = topk1_2d.detach().cpu()

            self.calculate_metrics(topk3_2d, label, ref_item_ids, metrics_2d, k=3)
            self.calculate_metrics(topk2_2d, label, ref_item_ids, metrics_2d, k=2)
            self.calculate_metrics(topk1_2d, label, ref_item_ids, metrics_2d, k=1)
            original_score = self._calculate_top_score(metrics_2d)

            self.calculate_metrics(topk3_2d, label, ref_item_ids, metrics_2d_score, k=3)
            self.calculate_metrics(topk2_2d, label, ref_item_ids, metrics_2d_score, k=2)
            self.calculate_metrics(topk1_2d, label, ref_item_ids, metrics_2d_score, k=1)

            candidate_len = 16
            batch_size = 16
            total_batches = (candidate_len + batch_size - 1) // batch_size

            pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
            query_image_path = query_dataloader.dataset._get_image_path(pick_id)
            query_points0 = get_query_points(query_image_path[0])  #[N, 2]  (N=20)
            query_points1 = get_query_points(query_image_path[1])  #[N, 2]  (N=20)
            query_points2 = get_query_points(query_image_path[2])  #[N, 2]  (N=20)
            conf_per_pick = []
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                batch_image_list0 = []
                batch_image_list1 = []
                batch_image_list2 = []
                #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                #ref_image_paths = []
                #for k in range(len(ref_ids)):
                #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                for index in topk16_iids[i][start_idx:end_idx].tolist():
                    image_path0 = [query_image_path[0]]
                    image_path1 = [query_image_path[1]]
                    image_path2 = [query_image_path[2]]
                    ref_image_path = ref_dataloader.dataset._get_image_path(index)
                    image_path0.extend([ref_image_path])
                    image_path1.extend([ref_image_path])
                    image_path2.extend([ref_image_path])
                    images0 = load_and_preprocess_images(image_path0, mode="pad")  #[2, C, H, W]
                    images1 = load_and_preprocess_images(image_path1, mode="pad")  #[2, C, H, W]
                    images2 = load_and_preprocess_images(image_path2, mode="pad")  #[2, C, H, W]
                    batch_image_list0.append(images0)
                    batch_image_list1.append(images1)
                    batch_image_list2.append(images2)
                batch_images0 = torch.stack(batch_image_list0, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_images1 = torch.stack(batch_image_list1, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_images2 = torch.stack(batch_image_list2, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                batch_query_points0 = query_points0.repeat(batch_images0.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                batch_query_points1 = query_points1.repeat(batch_images1.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                batch_query_points2 = query_points2.repeat(batch_images2.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                predictions0 = self.vggt.forward(batch_images0, batch_query_points0)
                batch_conf0 = predictions0["conf"]  #[B, 2, N]
                conf0 = torch.mean(batch_conf0[:, 1, :], dim=-1, keepdim=False)  #[B]

                predictions1 = self.vggt.forward(batch_images1, batch_query_points1)
                batch_conf1 = predictions1["conf"]  #[B, 2, N]
                conf1 = torch.mean(batch_conf1[:, 1, :], dim=-1, keepdim=False)  #[B]

                predictions2 = self.vggt.forward(batch_images2, batch_query_points2)
                batch_conf2 = predictions2["conf"]  #[B, 2, N]
                conf2 = torch.mean(batch_conf2[:, 1, :], dim=-1, keepdim=False)  #[B]

                conf = (conf0 + conf1 + conf2) / 3.0
                conf_per_pick.append(conf)
            conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
            #conf_per_pick = self.threshold_process(conf_per_pick)
            #print(f"conf_per_pick:{conf_per_pick}")

            topk3 = conf_per_pick.topk(3, dim=0)
            topk2 = conf_per_pick.topk(2, dim=0)
            topk1 = conf_per_pick.topk(1, dim=0)

            topk3_iids = topk16_iids[i][topk3.indices]
            topk2_iids = topk16_iids[i][topk2.indices]
            topk1_iids = topk16_iids[i][topk1.indices]

            topk3_iids = topk3_iids.detach().cpu()
            topk2_iids = topk2_iids.detach().cpu()
            topk1_iids = topk1_iids.detach().cpu()

            self.calculate_metrics(topk3_iids, label, ref_item_ids, metrics_rerank, k=3)
            self.calculate_metrics(topk2_iids, label, ref_item_ids, metrics_rerank, k=2)
            self.calculate_metrics(topk1_iids, label, ref_item_ids, metrics_rerank, k=1)
            rerank_score = self._calculate_top_score(metrics_rerank)

            self.calculate_metrics(topk3_iids, label, ref_item_ids, metrics_rerank_score, k=3)
            self.calculate_metrics(topk2_iids, label, ref_item_ids, metrics_rerank_score, k=2)
            self.calculate_metrics(topk1_iids, label, ref_item_ids, metrics_rerank_score, k=1)

            score_diff = rerank_score - original_score
            label_rerank = 1 if score_diff > self.delta else 0
            self.label_list.append(label_rerank)

            if label_rerank == 1:
                positive_samples += 1
                #self.label_list.append([1, 0])
            else:
                negative_samples += 1
                #self.label_list.append([0, 1])

        self._save_samples(ref_dataloader.dataset.data_path, query_cls_feats)
        print(f"  positive samples: {positive_samples}")
        print(f"  negative samples: {negative_samples}")
        rerank_tr_r1 = metrics_rerank_score["recall"][1] / len(labels)
        rerank_tr_r2 = metrics_rerank_score["recall"][2] / len(labels)
        rerank_tr_r3 = metrics_rerank_score["recall"][3] / len(labels)

        base_tr_r1 = metrics_2d_score["recall"][1] / len(labels)
        base_tr_r2 = metrics_2d_score["recall"][2] / len(labels)
        base_tr_r3 = metrics_2d_score["recall"][3] / len(labels)

        eval_result = {
            "rerank_tr_r1": rerank_tr_r1 * 100.0,
            "rerank_tr_r2": rerank_tr_r2 * 100.0,
            "rerank_tr_r3": rerank_tr_r3 * 100.0,

            "rerank_average_score": (rerank_tr_r1 + rerank_tr_r2 + rerank_tr_r3 ) / 3.0, 

            "base_tr_r1": base_tr_r1 * 100.0,
            "base_tr_r2": base_tr_r2 * 100.0,
            "base_tr_r3": base_tr_r3 * 100.0,

            "base_average_score": (base_tr_r1 + base_tr_r2 + base_tr_r3) / 3.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "rerank_average_score"
        

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandler(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None


    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k(self, topk, labels, ref_dataloader,  k=10):
        correct = 0.0
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            if labels[i] in predict_label:
                correct += 1.0

        return correct / len(labels)

    def calculate_accuracy_at_k(self, topk, labels, ref_dataloader, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            predict_label = Counter(predict_label).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)

        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).cpu()
        refids = torch.LongTensor(refids).cpu()
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        # select score base on pickids and ref gallary ids
        pick_ids =  query_dataloader.dataset._get_pick_id(pickids.tolist()) # real pick ids in string
        per_pick_ref_ids = ref_dataloader.dataset._get_per_pick_ref_ids(pick_ids)
        # per_pick_ref_ids = torch.as_tensor(per_pick_ref_ids).to(scores.device)
        # per_pick_ref_ids = torch.LongTensor(per_pick_ref_ids).to(scores.device)
        per_pick_ref_ids = np.array(per_pick_ref_ids, dtype=object)
        # select score base on pickids and ref gallary ids
        # per_pick_ref_ids = torch.tensor([1,2,3,6,10,12])

        all_topk5_iids = []
        all_topk3_iids = []
        all_topk2_iids = []
        all_topk1_iids = []

        for i in range(len(scores)):
            reduced_scores = scores[i][per_pick_ref_ids[i]]
            # reduced_scores = scores[i,[23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
            topk5 = reduced_scores.topk(5, dim=0)
            topk3 = reduced_scores.topk(3, dim=0)
            topk2 = reduced_scores.topk(2, dim=0)
            topk1 = reduced_scores.topk(1, dim=0)

            per_pick_ref_id = np.array(per_pick_ref_ids[i])
            # topk10_iids = refids[topk10.indices]
            topk5_iids = per_pick_ref_id[topk5.indices.tolist()]
            topk3_iids = per_pick_ref_id[topk3.indices.tolist()]
            topk2_iids = per_pick_ref_id[topk2.indices.tolist()]
            topk1_iids = per_pick_ref_id[topk1.indices.tolist()]

            topk5_iids = topk5_iids
            topk3_iids = topk3_iids
            topk2_iids = topk2_iids
            topk1_iids = topk1_iids

            all_topk5_iids.append(topk5_iids)
            all_topk3_iids.append(topk3_iids)
            all_topk2_iids.append(topk2_iids)
            all_topk1_iids.append(topk1_iids)



        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        tr_r3 = self.calculate_recall_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        tr_r2 = self.calculate_recall_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        tr_r1 = self.calculate_recall_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        acc_r3 = self.calculate_accuracy_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        acc_r2 = self.calculate_accuracy_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        acc_r1 = self.calculate_accuracy_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandlerVGGT(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.correct_recall_dict = {}
        self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.classifier = RerankClassifier()
        self.vggt.to("cuda").eval()
        self.classifier.to("cuda").eval()


    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k(self, topk, labels, ref_dataloader, k=10):
        correct = 0.0
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            if labels[i] in predict_label:
                correct += 1.0

        return correct / len(labels)

    def calculate_accuracy_at_k(self, topk, labels, ref_dataloader, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            predict_label = Counter(predict_label).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)

        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).cpu()
        refids = torch.LongTensor(refids).cpu()
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        # select score base on pickids and ref gallary ids
        pick_ids =  query_dataloader.dataset._get_pick_id(pickids.tolist()) # real pick ids in string
        per_pick_ref_ids = ref_dataloader.dataset._get_per_pick_ref_ids(pick_ids)
        # per_pick_ref_ids = torch.as_tensor(per_pick_ref_ids).to(scores.device)
        # per_pick_ref_ids = torch.LongTensor(per_pick_ref_ids).to(scores.device)
        per_pick_ref_ids = np.array(per_pick_ref_ids, dtype=object)
        # select score base on pickids and ref gallary ids
        # per_pick_ref_ids = torch.tensor([1,2,3,6,10,12])

        all_topk5_iids = []
        all_topk3_iids = []
        all_topk2_iids = []
        all_topk1_iids = []

        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size

        for i in tqdm(range(len(scores)), desc='VGGT RERANKING'):
            per_pick_ref_id = torch.LongTensor(per_pick_ref_ids[i])
            reduced_scores = scores[i][per_pick_ref_id]
            # reduced_scores = scores[i,[23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
            topk16 = reduced_scores.topk(16, dim=0)  #[16]
            topk16_iids = per_pick_ref_id[topk16.indices].to("cuda")
            query_feat = query_cls_feats[i].unsqueeze(0).to('cuda')
            logits = self.classifier(features=query_feat, only_infer=True)
            predict = torch.argmax(logits, dim=1)
            if predict.item() == 0:
                topk5_iids = topk16_iids[:5]
                topk3_iids = topk16_iids[:3]
                topk2_iids = topk16_iids[:2]
                topk1_iids = topk16_iids[:1]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()
            else:
                pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
                query_image_path = query_dataloader.dataset._get_image_path(pick_id)
                query_points = get_query_points(query_image_path)  #[N, 2]  (N=20)
                conf_per_pick = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                    batch_image_list = []
                    #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                    #ref_image_paths = []
                    #for k in range(len(ref_ids)):
                    #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                    #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                    for index in topk16_iids[start_idx:end_idx].tolist():
                        image_path = [query_image_path]
                        ref_image_path = ref_dataloader.dataset._get_image_path(index)
                        image_path.extend([ref_image_path])
                        images = load_and_preprocess_images(image_path, mode="pad")  #[2, C, H, W]
                        batch_image_list.append(images)
                    batch_images = torch.stack(batch_image_list, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_query_points = query_points.repeat(batch_images.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    with torch.amp.autocast("cuda"):
                        predictions = self.vggt.forward(batch_images, batch_query_points)
                    batch_conf = predictions["conf"]  #[B, 2, N]
                    conf = torch.mean(batch_conf[:, 1, :], dim=-1, keepdim=False)  #[B]
                    conf_per_pick.append(conf)
                conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
                #conf_per_pick = self.threshold_process(conf_per_pick)

                topk5 = conf_per_pick.topk(5, dim=0)
                topk3 = conf_per_pick.topk(3, dim=0)
                topk2 = conf_per_pick.topk(2, dim=0)
                topk1 = conf_per_pick.topk(1, dim=0)

                topk5_iids = topk16_iids[topk5.indices]
                topk3_iids = topk16_iids[topk3.indices]
                topk2_iids = topk16_iids[topk2.indices]
                topk1_iids = topk16_iids[topk1.indices]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()

            all_topk5_iids.append(topk5_iids)
            all_topk3_iids.append(topk3_iids)
            all_topk2_iids.append(topk2_iids)
            all_topk1_iids.append(topk1_iids)

        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        tr_r3 = self.calculate_recall_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        tr_r2 = self.calculate_recall_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        tr_r1 = self.calculate_recall_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        acc_r3 = self.calculate_accuracy_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        acc_r2 = self.calculate_accuracy_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        acc_r1 = self.calculate_accuracy_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchHandlerVGGT3t1(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        self.correct_recall_dict = {}
        self.correct_accuracy_dict = {}
        self.vggt = VGGT_adapter()
        self.classifier = RerankClassifier()
        self.vggt.to("cuda").eval()
        self.classifier.to("cuda").eval()


    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k(self, topk, labels, ref_dataloader, k=10):
        correct = 0.0
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            if labels[i] in predict_label:
                correct += 1.0

        return correct / len(labels)

    def calculate_accuracy_at_k(self, topk, labels, ref_dataloader, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = ref_dataloader.dataset._get_item_id(topk[i].tolist())
            predict_label = Counter(predict_label).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)

        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).cpu()
        refids = torch.LongTensor(refids).cpu()
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        # select score base on pickids and ref gallary ids
        pick_ids =  query_dataloader.dataset._get_pick_id(pickids.tolist()) # real pick ids in string
        per_pick_ref_ids = ref_dataloader.dataset._get_per_pick_ref_ids(pick_ids)
        # per_pick_ref_ids = torch.as_tensor(per_pick_ref_ids).to(scores.device)
        # per_pick_ref_ids = torch.LongTensor(per_pick_ref_ids).to(scores.device)
        per_pick_ref_ids = np.array(per_pick_ref_ids, dtype=object)
        # select score base on pickids and ref gallary ids
        # per_pick_ref_ids = torch.tensor([1,2,3,6,10,12])

        all_topk5_iids = []
        all_topk3_iids = []
        all_topk2_iids = []
        all_topk1_iids = []

        candidate_len = 16
        batch_size = 16
        total_batches = (candidate_len + batch_size - 1) // batch_size

        for i in tqdm(range(len(scores)), desc='VGGT RERANKING'):
            per_pick_ref_id = torch.LongTensor(per_pick_ref_ids[i])
            reduced_scores = scores[i][per_pick_ref_id]
            # reduced_scores = scores[i,[23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
            topk16 = reduced_scores.topk(16, dim=0)  #[16]
            topk16_iids = per_pick_ref_id[topk16.indices].to("cuda")
            query_feat = query_cls_feats[i].unsqueeze(0).to('cuda')
            logits = self.classifier(features=query_feat, only_infer=True)
            predict = torch.argmax(logits, dim=1)
            if predict.item() == 0:
                topk5_iids = topk16_iids[:5]
                topk3_iids = topk16_iids[:3]
                topk2_iids = topk16_iids[:2]
                topk1_iids = topk16_iids[:1]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()
            else:
                pick_id = query_dataloader.dataset.pick_ids[pickids[i].item()]
                query_image_path = query_dataloader.dataset._get_image_path(pick_id)
                query_points0 = get_query_points(query_image_path[0])  #[N, 2]  (N=20)
                query_points1 = get_query_points(query_image_path[1])  #[N, 2]  (N=20)
                query_points2 = get_query_points(query_image_path[2])  #[N, 2]  (N=20)
                conf_per_pick = []
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, candidate_len)
                    batch_image_list0 = []
                    batch_image_list1 = []
                    batch_image_list2 = []
                    #ref_ids = [ref_item_ids[index] for index in topk100_iids[i][start_idx:end_idx].tolist()]
                    #ref_image_paths = []
                    #for k in range(len(ref_ids)):
                    #    for path in ref_dataloader.dataset.items[ref_ids[k]]:
                    #        ref_image_paths = ref_image_paths.extend(os.path.join(ref_dataloader.dataset.data_path, 'Reference_Images', ref_ids[k], path))
                    for index in topk16_iids[start_idx:end_idx].tolist():
                        image_path0 = [query_image_path[0]]
                        image_path1 = [query_image_path[1]]
                        image_path2 = [query_image_path[2]]
                        ref_image_path = ref_dataloader.dataset._get_image_path(index)
                        image_path0.extend([ref_image_path])
                        image_path1.extend([ref_image_path])
                        image_path2.extend([ref_image_path])
                        images0 = load_and_preprocess_images(image_path0, mode="pad")  #[2, C, H, W]
                        images1 = load_and_preprocess_images(image_path1, mode="pad")  #[2, C, H, W]
                        images2 = load_and_preprocess_images(image_path2, mode="pad")  #[2, C, H, W]
                        batch_image_list0.append(images0)
                        batch_image_list1.append(images1)
                        batch_image_list2.append(images2)
                    batch_images0 = torch.stack(batch_image_list0, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_images1 = torch.stack(batch_image_list1, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_images2 = torch.stack(batch_image_list2, dim=0).float().to("cuda")  #[B, 2, C, H, W]
                    batch_query_points0 = query_points0.repeat(batch_images0.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    batch_query_points1 = query_points1.repeat(batch_images1.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    batch_query_points2 = query_points2.repeat(batch_images2.size(0), 1, 1).float().to("cuda")   #[B, N, 2]
                    predictions0 = self.vggt.forward(batch_images0, batch_query_points0)
                    batch_conf0 = predictions0["conf"]  #[B, 2, N]
                    conf0 = torch.mean(batch_conf0[:, 1, :], dim=-1, keepdim=False)  #[B]

                    predictions1 = self.vggt.forward(batch_images1, batch_query_points1)
                    batch_conf1 = predictions1["conf"]  #[B, 2, N]
                    conf1 = torch.mean(batch_conf1[:, 1, :], dim=-1, keepdim=False)  #[B]

                    predictions2 = self.vggt.forward(batch_images2, batch_query_points2)
                    batch_conf2 = predictions2["conf"]  #[B, 2, N]
                    conf2 = torch.mean(batch_conf2[:, 1, :], dim=-1, keepdim=False)  #[B]

                    conf = (conf0 + conf1 + conf2) / 3.0
                    conf_per_pick.append(conf)
                conf_per_pick = torch.cat(conf_per_pick, dim=0)  #[16]
                #conf_per_pick = self.threshold_process(conf_per_pick)

                topk5 = conf_per_pick.topk(5, dim=0)
                topk3 = conf_per_pick.topk(3, dim=0)
                topk2 = conf_per_pick.topk(2, dim=0)
                topk1 = conf_per_pick.topk(1, dim=0)

                topk5_iids = topk16_iids[topk5.indices]
                topk3_iids = topk16_iids[topk3.indices]
                topk2_iids = topk16_iids[topk2.indices]
                topk1_iids = topk16_iids[topk1.indices]

                topk5_iids = topk5_iids.detach().cpu()
                topk3_iids = topk3_iids.detach().cpu()
                topk2_iids = topk2_iids.detach().cpu()
                topk1_iids = topk1_iids.detach().cpu()

            all_topk5_iids.append(topk5_iids)
            all_topk3_iids.append(topk3_iids)
            all_topk2_iids.append(topk2_iids)
            all_topk1_iids.append(topk1_iids)

        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        tr_r3 = self.calculate_recall_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        tr_r2 = self.calculate_recall_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        tr_r1 = self.calculate_recall_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(all_topk5_iids, labels, ref_dataloader, k=5)
        acc_r3 = self.calculate_accuracy_at_k(all_topk3_iids, labels, ref_dataloader, k=3)
        acc_r2 = self.calculate_accuracy_at_k(all_topk2_iids, labels, ref_dataloader, k=2)
        acc_r1 = self.calculate_accuracy_at_k(all_topk1_iids, labels, ref_dataloader, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())


class Armbench3t1Handler(ArmbenchHandlerAllrefVGGT3t1):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        # with open(os.path.join(data_path, 'Picks_splits', 'train_label_dict.json',  'r')) as f:
        #     self.train_labels = json.load(f)
        #
        # with open(os.path.join(data_path, 'Picks_splits', 'test_label_dict.json',  'r')) as f:
        #     self.test_labels = json.load(f)


    def train_batch(self, model, image0, image1, image2, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        query_images = [image0, image1, image2]
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }



    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image0=None, image1=None, image2=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if image0 is not None:
            query_images = [image0, image1, image2]
        else:
            query_images = None

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())



    # def build_rank(self, data_loader, topk, values, query_data_type, k=1000):
    #     all_rank = []
    #     topk = topk.detach().cpu() #.reshape(-1, k)
    #     values = values.detach().cpu() #.reshape(-1, k)
    #
    #     if query_data_type == 'img':
    #         retrieval_type = 'text'
    #     elif query_data_type == 'text':
    #         retrieval_type = 'img'
    #
    #     print(f"Build rank list for {query_data_type} to {retrieval_type}")
    #
    #     for idx in tqdm(range(topk.shape[0])):
    #         if query_data_type == 'img':
    #             item_id = data_loader.dataset._get_img_id(idx)
    #         elif query_data_type == 'text':
    #             item_id = data_loader.dataset._get_text_id(idx)
    #
    #         rank_list = topk[idx].tolist()
    #         # transfer rank idx to item id
    #         if retrieval_type == 'img':
    #             rank_list = data_loader.dataset._get_img_id(rank_list)
    #         elif retrieval_type == 'text':
    #             rank_list = data_loader.dataset._get_text_id(rank_list)
    #
    #         all_rank.append({'query_id': item_id,
    #                         'rank': rank_list,
    #                         'scores': values[idx].tolist()})
    #
    #     return all_rank
    #
    #
    # def calculate_recall_at_k(self, topk, labels, ref_item_ids, k=10):
    #     correct = 0.0
    #     for i in range(len(labels)):
    #         if labels[i] in [ref_item_ids[index] for index in topk[i].tolist()]:
    #             correct += 1.0
    #
    #     return correct / len(labels)
    #
    # def calculate_accuracy_at_k(self, topk, labels, ref_item_ids, k=10):
    #     correct = 0.0
    #     from collections import Counter
    #     for i in range(len(labels)):
    #         predict_label = Counter([ref_item_ids[index] for index in topk[i].tolist()]).most_common(1)[0][0]
    #         if predict_label == labels[i]:
    #             correct += 1.0
    #
    #     return correct / len(labels)
    #
    #
    #
    # def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):
    #
    #     # query_image_feats = {}
    #     # for feats, ids in zip(self.query_image_feats, self.pick_ids):
    #     #     for i, _idx in enumerate(ids):
    #     #         idx = _idx.item()
    #     #         if idx not in query_image_feats:
    #     #             query_image_feats[idx] = feats[i]
    #     #
    #     pickids = torch.cat(self.pick_ids, dim=0)
    #     refids = torch.cat(self.ref_ids, dim=0)
    #     # iids = []
    #     # sorted_tensors = []
    #     # for key in sorted(query_image_feats.keys()):
    #     #     sorted_tensors.append(query_image_feats[key].view(1, -1))
    #     #     iids.append(key)
    #     #
    #     query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0)) #.to('cuda')
    #     ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0)) #.to('cuda')
    #
    #     labels = query_dataloader.dataset._get_class_id(pickids.tolist())
    #     ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())
    #
    #     scores = query_cls_feats @ ref_cls_feats.t()
    #
    #
    #     pickids = torch.LongTensor(pickids).to(scores.device)
    #     refids = torch.LongTensor(refids).to(scores.device)
    #     # labels = labels.to(scores.device)
    #     # ref_item_ids = ref_item_ids.to(scores.device)
    #
    #     # iids = torch.LongTensor(iids).to(scores.device)
    #
    #     print("scores: {}".format(scores.size()))
    #     print("pickids: {}".format(pickids.size()))
    #     print("refids: {}".format(refids.size()))
    #
    #
    #
    #     # topk10 = scores.topk(10, dim=1)
    #     topk5 = scores.topk(5, dim=1)
    #     topk3 = scores.topk(3, dim=1)
    #     topk2 = scores.topk(2, dim=1)
    #     topk1 = scores.topk(1, dim=1)
    #
    #     # topk10_iids = refids[topk10.indices]
    #     topk5_iids = refids[topk5.indices]
    #     topk3_iids = refids[topk3.indices]
    #     topk2_iids = refids[topk2.indices]
    #     topk1_iids = refids[topk1.indices]
    #
    #     # topk10_iids = topk10_iids.detach().cpu()
    #     topk5_iids = topk5_iids.detach().cpu()
    #     topk3_iids = topk3_iids.detach().cpu()
    #     topk2_iids = topk2_iids.detach().cpu()
    #     topk1_iids = topk1_iids.detach().cpu()
    #
    #     # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
    #     tr_r5 = self.calculate_recall_at_k(topk5_iids, labels, ref_item_ids, k=5)
    #     tr_r3 = self.calculate_recall_at_k(topk3_iids, labels, ref_item_ids, k=3)
    #     tr_r2 = self.calculate_recall_at_k(topk2_iids, labels, ref_item_ids, k=2)
    #     tr_r1 = self.calculate_recall_at_k(topk1_iids, labels, ref_item_ids, k=1)
    #
    #     # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
    #     acc_r5 = self.calculate_accuracy_at_k(topk5_iids, labels, ref_item_ids, k=5)
    #     acc_r3 = self.calculate_accuracy_at_k(topk3_iids, labels, ref_item_ids, k=3)
    #     acc_r2 = self.calculate_accuracy_at_k(topk2_iids, labels, ref_item_ids, k=2)
    #     acc_r1 = self.calculate_accuracy_at_k(topk1_iids, labels, ref_item_ids, k=1)
    #
    #     eval_result = {
    #         "tr_r1": tr_r1 * 100.0,
    #         "tr_r2": tr_r2 * 100.0,
    #         "tr_r3": tr_r3 * 100.0,
    #         "tr_r5": tr_r5 * 100.0,
    #
    #         # "tr_r10": tr_r10 * 100.0,
    #
    #         "acc_r1": acc_r1 * 100.0,
    #         "acc_r2": acc_r2 * 100.0,
    #         "acc_r3": acc_r3 * 100.0,
    #         "acc_r5": acc_r5 * 100.0,
    #         # "acc_r10": acc_r10 * 100.0,
    #
    #         "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
    #     }
    #
    #     print('* Eval result = %s' % json.dumps(eval_result))
    #     return eval_result, "average_score"
    #
    #     # if build_ranking:
    #     #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
    #     #
    #     # else:
    #     #     return eval_result, "average_score"
    #

class Armbench3t1extractHandler(ArmbenchHandlerAllrefVGGT3t1extract):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        # with open(os.path.join(data_path, 'Picks_splits', 'train_label_dict.json',  'r')) as f:
        #     self.train_labels = json.load(f)
        #
        # with open(os.path.join(data_path, 'Picks_splits', 'test_label_dict.json',  'r')) as f:
        #     self.test_labels = json.load(f)


    def train_batch(self, model, image0, image1, image2, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        query_images = [image0, image1, image2]
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }



    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image0=None, image1=None, image2=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if image0 is not None:
            query_images = [image0, image1, image2]
        else:
            query_images = None

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class Armbench3t1sampleHandler(ArmbenchHandlerAllrefVGGT3t1sample):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        # with open(os.path.join(data_path, 'Picks_splits', 'train_label_dict.json',  'r')) as f:
        #     self.train_labels = json.load(f)
        #
        # with open(os.path.join(data_path, 'Picks_splits', 'test_label_dict.json',  'r')) as f:
        #     self.test_labels = json.load(f)


    def train_batch(self, model, image0, image1, image2, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        query_images = [image0, image1, image2]
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }



    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image0=None, image1=None, image2=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if image0 is not None:
            query_images = [image0, image1, image2]
        else:
            query_images = None

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

class ArmbenchPick1Handler(ArmbenchHandlerAllrefVGGT):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }

class ArmbenchPick1extractHandler(ArmbenchHandlerAllrefVGGTextract):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }
    
class ArmbenchPick1sampleHandler(ArmbenchHandlerAllrefVGGTsample):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }

class ArmbenchPick1t1Handler(ArmbenchHandlerAllref):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }

    #def eval_batch(self, model, pick_image_vggt=None, ref_image_vggt=None, pick_id=None, ref_id=None, padding_mask=None):
    #    if pick_image_vggt is not None and ref_image_vggt is not None:
    #        query_vision_cls, ref_vision_cls = model(query_images_vggt=pick_image_vggt, ref_image_vggt=ref_image_vggt, only_infer=True)
    #        self.query_image_feats.append(query_vision_cls.detach().cpu())
    #        self.ref_image_feats.append(ref_vision_cls.detach().cpu())
    #        # self.text_feats.append(language_cls.clone())
    #        self.pick_ids.append(pick_id.detach().cpu())
    #        self.ref_ids.append(ref_id.detach().cpu())
    #
    #    elif pick_image_vggt is not None and ref_image_vggt is None:
    #        query_vision_cls, _ = model(query_images_vggt=pick_image_vggt, only_infer=True)
    #        self.query_image_feats.append(query_vision_cls.detach().cpu())
    #        self.pick_ids.append(pick_id.detach().cpu())
    #
    #    elif pick_image_vggt is None and ref_image_vggt is not None:
    #        _, ref_vision_cls = model(ref_image_vggt=ref_image_vggt, only_infer=True)
    #        self.ref_image_feats.append(ref_vision_cls.detach().cpu())
    #        self.ref_ids.append(ref_id.detach().cpu())

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())


class ArmbenchPick1CLlossHandler(ArmbenchHandlerAllref):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, ref_id,
                    language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image, ref_id=ref_id)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }

class ArmbenchPick1NearestRefHandler(ArmbenchHandler):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_images, ref_id,
                    language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, = model(query_images=pick_image, ref_images=ref_images, ref_id=ref_id)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }



def get_handler(args):
    if args.task == "armbench3t1":
        return Armbench3t1Handler()
    elif args.task == "armbench3t1extract":
        return Armbench3t1extractHandler()
    elif args.task == "armbench3t1sample":
        return Armbench3t1sampleHandler()
    elif args.task == "armbenchpick1":
        return ArmbenchPick1Handler()
    elif args.task == "armbenchpick1extract":
        return ArmbenchPick1extractHandler()
    elif args.task == "armbenchpick1sample":
        return ArmbenchPick1sampleHandler()
    elif args.task == "armbenchpick1to1":
        return ArmbenchPick1t1Handler()
    elif args.task == "armbenchpick1_clloss":
        return ArmbenchPick1CLlossHandler()
    elif args.task == "armbenchpick1_nearestref":
        return ArmbenchPick1NearestRefHandler()
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)




def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            if not isinstance(data[tensor_key], list):
                # data[tensor_key] = [t.to(device, non_blocking=True) for t in data[tensor_key]]
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and 'image' in tensor_key:
                data[tensor_key] = data[tensor_key].half()

        # pick_images = [data['image0'], data['image1'], data['image2']]
        #
        # if loss_scaler is None:
        #     results = handler.train_batch(model, query_images=pick_images,
        #                                   ref_image=data["ref_image"], pick_id=data["pick_id"])
        # else:
        #     with torch.cuda.amp.autocast():
        #         results = handler.train_batch(model, query_images=pick_images,
        #                                   ref_image=data["ref_image"], pick_id=data["pick_id"])
        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.amp.autocast("cuda"):
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(query_dataloader, answer_dataloader, model, device, handler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # prepare query data
    # if args.retrieval_mode == 'text_to_image':

    # elif args.retrieval_mode == 'image_to_text':
    #     querys = load_dataset(args.dataset_url, split='train',
    #                                  num_proc=4)

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger)

    # build query embeddings

    for data in tqdm(query_dataloader):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        #     images = [data['image0'], data['image1'], data['image2']]
        #     handler.eval_batch(model=model, query_images=images, pick_id=data["pick_id"])
        with torch.amp.autocast("cuda"):
            handler.eval_batch(model=model, **data)

    # if args.load_embeddings_from_npy:
    #     raise NotImplementedError
        # # print(sorted(os.listdir(args.embeddings_file_path)))
        # freq = 3000
        # paths = [f'image_feats_{pointer}_freq_3000_gpu_0.npy' for pointer in range(0, 168001, freq)]
        # for file in paths:
        #     if file.endswith('.npy'):
        #         handler.query_image_feats.append(np.load(os.path.join(args.embeddings_file_path, file)))
        #
        # handler.query_image_feats = np.concatenate(handler.query_image_feats, axis=0)
        # handler.query_image_feats = torch.from_numpy(handler.query_image_feats)
        # handler.pick_ids = torch.arange(handler.query_image_feats.shape[0])
    # else:
    for data in tqdm(answer_dataloader):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            handler.eval_batch(model=model, **data)

            # if len(handler.query_image_feats) % handler.store_feq == 0:
            #     if args.dist_eval:
            #         handler.store_feats(mode='image', tag=args.model+'_'+args.finetune.split('/')[-1], gpu_id= args.gpu)
            #     else:
            #         handler.store_feats(mode='image', tag=args.model+'_'+args.finetune.split('/')[-1], gpu_id= 0)
            #     handler.store_pointer += handler.store_feq


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return handler.after_eval(query_dataloader, answer_dataloader, args)
