"""
Refactored CLIP-Adapter implementation with enhanced flexibility for expansion.

This module is a refactored and extended version of the original CLIP-Adapter
project by Gao Peng (https://github.com/gaopengcuhk/CLIP-Adapter), designed to
support additional datasets and more flexible hardware usage.

Original source repository:
https://github.com/gaopengcuhk/CLIP-Adapter
"""

import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from sklearn.metrics import f1_score
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# Templates to guide text prompt generation for each dataset
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",

    # GlobalStreetScapes templates:
    # These templates describe street-level photos with specific conditions.
    "GlobalStreetScapes_Weather": "a photo in {} weather.",
    "GlobalStreetScapes_Lighting": "a photo taken during {}.", # dawn/dusk -> twilight
    "GlobalStreetScapes_Glare": "a photo with {}." # yes -> glare; no -> no glare
}


def load_clip_model(cfg):
    """
    Load the CLIP model onto GPU if available, otherwise CPU.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model, preprocess = clip.load(backbone_name, device=device)
    
    return model, device


class Adapter(nn.Module):
    """
    A lightweight Adapter module that projects image features through
    a small bottleneck MLP to allow fine-tuning with minimal parameters.
    """
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):
    """
    Encodes classnames into CLIP text embeddings using predefined templates.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def normalize_classnames(self):
        """
        Normalize classnames based on dataset-specific rules.
        Returns a list of normalized class names (strings).
        """
        name = self.cfg.DATASET.NAME
        normalized = []

        for c in self.classnames:
            # Ensure c is a string
            c_str = str(c).replace("_", " ").lower()

            if name == "GlobalStreetScapes_Lighting":
                if c_str in ["dawn", "dusk", "dawn/dusk", "dusk/dawn"]:
                    normalized.append("twilight")
                else:
                    normalized.append(c_str)

            elif name == "GlobalStreetScapes_Glare":
                if c_str == "yes":
                    normalized.append("glare")
                elif c_str == "no":
                    normalized.append("no glare")
                else:
                    normalized.append(c_str)

            else:
                # Default: use original string
                normalized.append(c_str)

        return normalized

    
    def forward(self):
        # Get template for current dataset
        template = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]

        # Normalize classnames based on the dataset
        normalized_classnames = self.normalize_classnames()
        prompts = [template.format(c) for c in normalized_classnames]

        # Tokenize and encode prompts
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to("cuda")
        text_features = self.clip_model.encode_text(prompts)
        return text_features


class CustomCLIP(nn.Module):
    """
    A custom CLIP model that includes an image adapter for fine-tuning.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Use output_dim from the CLIP image encoder to match dimensionality
        self.adapter = Adapter(clip_model.visual.output_dim,
                               reduction=4).to(self.dtype)

            
    def forward(self, image):
        # Encode images and apply adapter
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        # Linearly blend the original and adapted features
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        # Encode text features
        text_features = self.text_encoder()

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity scaled by a learnable temperature
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_Adapter(TrainerX):
    """ CLIP-Adapter """

    def build_loss(self):
        """Build and initialize the loss function"""
        cfg = self.cfg
        lossname = cfg.TRAINER.LOSS.NAME

        if lossname == "ce":
            return nn.CrossEntropyLoss()
        
        elif lossname == "focal":
            gamma = cfg.TRAINER.LOSS.FOCAL_GAMMA
            alpha = cfg.TRAINER.LOSS.FOCAL_ALPHA

            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, alpha=None):
                    super().__init__()
                    self.gamma = gamma
                    self.alpha = alpha

                def forward(self, inputs, targets):
                    log_probs = F.log_softmax(inputs, dim=1)
                    probs = torch.exp(log_probs)
                    targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
                    pt = (probs * targets_onehot).sum(dim=1)

                    if self.alpha is not None:
                        alpha_t = self.alpha[targets]
                        loss = -alpha_t * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-9)
                    else:
                        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-9)

                    return loss.mean()

            alpha_tensor = None
            if alpha is not None:
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32).cuda()

            return FocalLoss(gamma=gamma, alpha=alpha_tensor)
        else:
            raise ValueError(f"Unknown loss name: {lossname}")

    def build_model(self):
        """Build and initialize the CLIP-Adapter model."""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, self.device = load_clip_model(cfg)
        clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.loss_fn = self.build_loss()

        # Build optimizer and scheduler for the adapter only
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("clip_adapter",
                            self.model.adapter, self.optim, self.sched)

    def forward_backward(self, batch):
        """
        Forward and backward pass for one training batch.
        Computes loss and updates parameters.
        """

        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = self.loss_fn(output, label)
        self.model_backward_and_update(loss)

        # Convert tensors to CPU numpy arrays for metric calculation
        output_cpu = output.detach().cpu().numpy()
        label_cpu = label.detach().cpu().numpy()
        
        # For F1 score, we need to convert probabilities to predicted class indices
        preds = output_cpu.argmax(axis=1)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "f1": f1_score(label_cpu, preds, average="macro", zero_division=0) * 100
        }

        # Update learning rate at the end of the epoch
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """
        Parse training batch and move data to the correct device.
        """
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        """
        Load model checkpoint from a directory. If epoch is not specified,
        loads the best model.
        """
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"Model not found at '{model_path}'")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f"Loading weights to {name} from '{model_path}' (epoch = {epoch})")

            # Allow partial loading by setting strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
