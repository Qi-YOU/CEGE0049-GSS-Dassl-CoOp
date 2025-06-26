"""
Global Street Scapes Dataset Loader for Dassl-CoOp

This module defines a base class `GlobalStreetScapesBase` for multi-attribute classification,
as well as three concrete subclasses for specific tasks:
- GlobalStreetScapes_Weather
- GlobalStreetScapes_Glare
- GlobalStreetScapes_Lighting

Each subclass handles a single attribute (weather, glare, lighting condition),
reads corresponding CSV files, and supports few-shot learning and class subsampling.

Expected dataset directory structure:
global_street_scapes/
├── img/
│   ├── 1/xxx.jpeg, /...
│   ├── 2/...
│   ├── 3/...
│   ├── 4/...
│   ├── 5/...
│   ├── 6/...
│   └── 7/...
├── train/
│   ├── weather.csv
│   ├── glare.csv
│   └── lighting_condition.csv
└── test/
    ├── weather.csv
    ├── glare.csv
    └── lighting_condition.csv
"""

import os
import csv
import math
import random
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


SUPPORTED_ATTR_NAMES = [
    "platform", "weather", "view_direction", "lighting_condition",
    "panoramic_status", "quality", "glare", "reflection"
]


class GlobalStreetScapesBase(DatasetBase):
    """
    Base class for Global Street Scapes dataset, supporting:
    - One attribute per subclass (e.g. weather, glare)
    - Few-shot sampling with NUM_SHOTS
    - Class subsampling with SUBSAMPLE_CLASSES

    Subclasses must set `attr_name`, e.g., 'weather'.
    """

    dataset_dir = "global_street_scapes"
    attr_name = None  # Must be defined by subclasses

    def __init__(self, cfg):
        assert self.attr_name is not None, "Subclasses must set `attr_name` for GlobalStreetScapes"
        assert self.attr_name in SUPPORTED_ATTR_NAMES, (
            f"attr_name '{self.attr_name}' is not supported. "
            f"Supported attributes: {SUPPORTED_ATTR_NAMES}"
        )

        # Setup paths
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Load data from CSVs
        train = self.read_data("train")
        test = self.read_data("test")

        # Split train into train/val
        train, val = self.split_trainval(train, p_val=0.2)

        # Apply few-shot sampling
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            random.seed(seed)
            train = self.generate_fewshot_dataset(train, num_shots)
            val = self.generate_fewshot_dataset(val, min(num_shots, 4))

        # Subsample base or new classes
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        # Print dataset statistics
        self.print_dataset_stats(train, val, test)

        # Final dataset assignment
        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        """
        Load data from CSV file for a given split ('train' or 'test').

        The CSV file contains a header. We locate the image path column and
        the attribute column (self.attr_name) dynamically based on the header.

        Args:
            split (str): "train" or "test"
        
        Returns:
            List of Datum objects with image path and attribute label.
        """
        label_path = os.path.join(self.dataset_dir, split, f"{self.attr_name}.csv")

        # First pass: collect unique labels
        unique_labels = set()

        with open(label_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            try:
                attr_col_idx = header.index(self.attr_name)
            except ValueError:
                raise ValueError(f"CSV {label_path} must have a '{self.attr_name}' column")

            for row in reader:
                label_str = row[attr_col_idx].strip().lower()
                unique_labels.add(label_str)

        # Build label to int mapping
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        if split == "train":
            print(f"Number of labels of {self.attr_name} class: {len(self.label_to_id)}\n")
            max_label_len = max(len(label) for label in self.label_to_id) # Compute max tabel length

            for label in sorted(self.label_to_id):
                print(f"- {label.ljust(max_label_len)}: {self.label_to_id[label]}") # Left-aligned output
            print()

        # Second pass: build dataset in datums
        items = []
        with open(label_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Find row index of correct colnames
            try:
                img_col_idx = header.index("img_path")  # Assuming the image path is listed as img_path
            except ValueError:
                raise ValueError(f"CSV {label_path} must have an 'img_path' column")

            try:
                attr_col_idx = header.index(self.attr_name)
            except ValueError:
                raise ValueError(f"CSV {label_path} must have a '{self.attr_name}' column")

            for row in reader:
                img_rel = row[img_col_idx]
                label_str = row[attr_col_idx].strip().lower()

                impath = os.path.join(self.dataset_dir, img_rel)
                label_int = self.label_to_id[label_str]

                # Generate the datum of path to the image,
                # label in 0-indexed integers,
                # and classnames in strings
                items.append(Datum(impath=impath, label=label_int, classname=label_str))

        return items

    @staticmethod
    def split_trainval(data, p_val=0.2):
        """
        Stratified split of training data into train/val.
        Ensures each class appears in both sets.
        """
        
        tracker = defaultdict(list)

        # Group indices of data samples by their class label
        for idx, item in enumerate(data):
            tracker[item.label].append(idx)

        train, val = [], []

        # For each class label, split the samples into val and train
        for label, idxs in tracker.items():

            # Calculate number of samples for validation set, at least 1
            n_val = max(1, round(len(idxs) * p_val))
            random.shuffle(idxs) # Shuffle indices to randomize split

            # Assign samples to val or train based on position after shuffling
            for i, idx in enumerate(idxs):
                # First n_val go to validation, rest go to training
                (val if i < n_val else train).append(data[idx])

        return train, val

    @staticmethod
    def generate_fewshot_dataset(data, num_shots):
        """
        Select up to `num_shots` samples per class from `data`.
        """
        
        label_to_items = defaultdict(list)

        # Collect all samples for each class label
        for item in data:
            label_to_items[item.label].append(item)

        output = []

        # For each class, randomly sample up to num_shots items
        for label, items in label_to_items.items():
            sampled = random.sample(items, min(num_shots, len(items)))
            output.extend(sampled) # Add sampled items to output list
        return output

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """
        Subsample classes to either "base" or "new".

        Classes are sorted alphabetically by string label, and divided into halves.
        Class labels are then re-indexed to be 0-based integers within the selected subset.
        """
        
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]

        # Extract unique class labels (strings) and sort them
        labels = sorted({item.label for item in dataset})  # string labels
        n = len(labels)
        m = math.ceil(n / 2) # Midpoint to split labels into two halves

        # Select either the first half or second half of labels based on subsample argument
        selected = labels[:m] if subsample == "base" else labels[m:]
        print(f"SUBSAMPLE {subsample.upper()} CLASSES: {len(selected)} out of {n}")

        # Map original string labels to new 0-based integer indices
        relabel = {label: idx for idx, label in enumerate(selected)}

        def filter_and_relabel(data):
            # Filter data to include only samples whose label is in selected
            # Also relabel their labels to new indices
            return [
                Datum(
                    impath=item.impath,
                    label=relabel[item.label], # New 0-based label
                    classname=item.classname
                )
                for item in data if item.label in selected
            ]

        # Apply filtering and relabeling for all provided datasets (train/val/test)
        return tuple(filter_and_relabel(data) for data in args)
    
    def print_dataset_stats(self, train, val, test):
        """
        Print formatted dataset statistics with sample counts and label distribution ratios.
        """
        # Format dataset name: replace underscores with space and title-case it
        class_name = self.__class__.__name__
        dataset_title = class_name.replace("_", " ").title()

        print(f"Dataset Statistics: {dataset_title}")

        for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
            class_counts = defaultdict(int)
            total_samples = len(split_data)

            # Count samples per class
            for item in split_data:
                class_counts[item.classname] += 1

            print(f"- {split_name} Set: {total_samples} samples")

            # Determine longest label for alignment
            if class_counts:
                max_label_len = max(len(label) for label in class_counts)
            else:
                max_label_len = 0

            # Print class counts and percentages
            for label in sorted(class_counts):
                count = class_counts[label]
                ratio = count / total_samples if total_samples > 0 else 0
                print(f"\t- {label.ljust(max_label_len)} : {count:4d} ({ratio:.2%})")
        print()

# Subclasses for each attribute
@DATASET_REGISTRY.register()
class GlobalStreetScapes_Platform(GlobalStreetScapesBase):
    """Global Street Scapes - Platform Attribute"""
    attr_name = "platform"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_Weather(GlobalStreetScapesBase):
    """Global Street Scapes - Weather Attribute"""
    attr_name = "weather"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_ViewDirection(GlobalStreetScapesBase):
    """Global Street Scapes - View Direction Attribute"""
    attr_name = "view_direction"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_LightingCondition(GlobalStreetScapesBase):
    """Global Street Scapes - Lighting Condition Attribute"""
    attr_name = "lighting_condition"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_PanoramicStatus(GlobalStreetScapesBase):
    """Global Street Scapes - Panoramic Status Attribute"""
    attr_name = "panoramic_status"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_Quality(GlobalStreetScapesBase):
    """Global Street Scapes - Quality Attribute"""
    attr_name = "quality"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_Glare(GlobalStreetScapesBase):
    """Global Street Scapes - Glare Attribute"""
    attr_name = "glare"

@DATASET_REGISTRY.register()
class GlobalStreetScapes_Reflection(GlobalStreetScapesBase):
    """Global Street Scapes - Reflection Attribute"""
    attr_name = "reflection"
