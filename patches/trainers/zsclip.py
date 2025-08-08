import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

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

    """
    Modified with minimal changes to support GlobalStreetScapes datasets.  
    """
    # GlobalStreetScapes templates:
    # These templates describe street-level photos with specific conditions.
    "GlobalStreetScapes_Platform": "a photo taken on {}.",  # e.g. driving/walking/slyching surface, railway, fields, tunnel
    "GlobalStreetScapes_Weather": "a photo in {} weather.",  # clear, cloudy, rainy, snowy, foggy
    "GlobalStreetScapes_ViewDirection": "a photo captured {} the road.",  # front/back -> along, side -> across
    "GlobalStreetScapes_LightingCondition": "a photo taken during {}.",  # day, night, dawn/dusk -> twilight
    "GlobalStreetScapes_PanoramicStatus": "a {} photo.",  # true -> panoramic; false -> non-panoramic
    "GlobalStreetScapes_Quality": "a photo of {} quality.",  # good, slightly poor, very poor
    "GlobalStreetScapes_Glare": "a photo with {}.",  # yes -> glare; no -> no glare
    "GlobalStreetScapes_Reflection": "a photo with {}."  # yes -> reflection; no -> no reflection
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    """
    Modified Zero-Shot functions with minimal changes to support GlobalStreetScapes datasets.
    """
    
    def normalize_classnames(self, classnames):
        """
        Normalize classnames based on dataset-specific rules.
        Returns a list of normalized class names (strings).

        Modified CLIP-Adapter functions with minimal changes to support GlobalStreetScans datasets.  
        """
        name = self.cfg.DATASET.NAME
        normalized = []

        for c in classnames:
            # Ensure c is a string
            c_str = str(c).replace("_", " ").lower()

            if name == "GlobalStreetScapes_LightingCondition":
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

            elif name == "GlobalStreetScapes_Reflection":
                if c_str == "yes":
                    normalized.append("reflection")
                elif c_str == "no":
                    normalized.append("no reflection")
                else:
                    normalized.append(c_str)

            elif name == "GlobalStreetScapes_ViewDirection":
                # front/back -> along, side -> across
                if c_str in ["front", "back", "front/back", "back/front"]:
                    normalized.append("along")
                elif c_str == "side":
                    normalized.append("across")
                else:
                    normalized.append(c_str)
            elif name == "GlobalStreetScapes_PanoramicStatus":
                # true -> ""; false -> "non-"
                if c_str in ["true", "yes", "panoramic"]:
                    normalized.append("panoramic")
                elif c_str in ["false", "no", "non-panoramic"]:
                    normalized.append("non-panoramic")
                else:
                    normalized.append(c_str)

            else:
                # Default: use original string
                normalized.append(c_str)

        return normalized

    def build_model(self):
        cfg = self.cfg
        classnames = self.normalize_classnames(self.dm.dataset.classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
