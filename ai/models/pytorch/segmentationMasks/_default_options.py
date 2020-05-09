'''
    Default configuration properties for PyTorch semantic segmentation models.
    May be overridden (also partially) by models subclassing the trainer,
    or else through the custom configuration loaded at runtime.

    2020 Benjamin Kellenberger
'''

DEFAULT_OPTIONS = {
    "general": {
        "image_size": [224, 224],
        "device": "cuda",
        "seed": 0
    },
    "model": {
        "kwargs": {
            "in_channels": 3,
            "depth": 5,
            "numFeaturesExponent": 6,
            "padding": False,
            "batch_norm": False,
            "upsamplingMode": "upconv"
        }
    },
    "dataset": {
        "class": "ai.models.pytorch.SegmentationDataset",
        "kwargs": {}
    },
    "train": {
        "dataLoader": {
            "kwargs": {
                "shuffle": True,
                "batch_size": 1
            }
        },
        "optim": {
            "class": "torch.optim.SGD",
            "kwargs": {
                "lr": 1e-4,
                "weight_decay": 0.0
            }
        },
        "transform": {
            "class": "ai.models.pytorch.segmentationMasks.Compose",
            "kwargs": {
                "transforms": [{
                        "class": "ai.models.pytorch.segmentationMasks.Resize",
                        "kwargs": {
                            "size": [224, 224]
                        }
                    },
                    {
                        "class": "ai.models.pytorch.segmentationMasks.RandomHorizontalFlip",
                        "kwargs": {
                            "p": 0.5
                        }
                    },
                    {
                        "class": "ai.models.pytorch.segmentationMasks.JointTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.ToTensor"
                            }
                        }
                    },
                    {
                        "class": "ai.models.pytorch.segmentationMasks.DefaultTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.Normalize",
                                "kwargs": {
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225]
                                }
                            }
                        }
                    }
                ]
            }
        },
        "criterion": {
            "class": "torch.nn.CrossEntropyLoss",
            "kwargs": {}
        }
    },
    "inference": {
        "transform": {
            "class": "ai.models.pytorch.segmentationMasks.Compose",
            "kwargs": {
                "transforms": [{
                        "class": "ai.models.pytorch.segmentationMasks.Resize",
                        "kwargs": {
                            "size": [224, 224]
                        }
                    },
                    {
                        "class": "ai.models.pytorch.segmentationMasks.JointTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.ToTensor"
                            }
                        }
                    },
                    {
                        "class": "ai.models.pytorch.segmentationMasks.DefaultTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.Normalize",
                                "kwargs": {
                                    "mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225]
                                }
                            }
                        }
                    }
                ]
            }
        },
        "dataLoader": {
            "kwargs": {
                "shuffle": False,
                "batch_size": 1
            }
        }
    }
}