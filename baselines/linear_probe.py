"""
Linear Probe for evaluating pretrained CLIP visual backbones via a simple linear classifier.

This module implements a standard linear probe training setup where a linear 
classification head is trained on frozen visual features extracted from a 
pretrained backbone. It serves as a baseline for assessing the quality of 
learned representations without fine-tuning the backbone itself.

The design follows conventional protocols widely used in representation learning research, 
such as in the seminal works on self-supervised learning and supervised pretraining.

Inspiration: https://github.com/openai/CLIP
Reference: Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PmLR, 2021.
"""

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class LinearProbeCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual  # frozen
        self.dtype = clip_model.dtype
        self.feature_dim = clip_model.ln_final.weight.shape[0]  # dimension of image feature

        # Freeze the CLIP visual encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.n_cls = len(classnames)
        # Linear classifier on top of frozen image features
        self.classifier = nn.Linear(self.feature_dim, self.n_cls)

    def forward(self, image):
        # Get image features (from the visual encoder)
        image_features = self.image_encoder(image.type(self.dtype))

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Linear logits
        logits = self.classifier(image_features.float())
        return logits


@TRAINER_REGISTRY.register()
class Linear_Probe(TrainerX):
    """Linear probe on top of frozen CLIP image features."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building Linear Probe model")
        self.model = LinearProbeCLIP(cfg, classnames, clip_model)

        print("Turning off gradients for CLIP visual encoder")
        # Only the classifier params require grad
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.classifier, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # optimizer only for classifier
        self.optim = build_optimizer(self.model.classifier, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("classifier", self.model.classifier, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} from {} (epoch={})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
