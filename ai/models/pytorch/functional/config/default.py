'''
    Default configuration entries for PyTorch models.
    May be overridden (also partially) by models subclassig the generic PyTorch trainer,
    or else through the custom configuration loaded at runtime.

    2019 Benjamin Kellenberger
'''

DEFAULT_OPTIONS = {
	"general": {
		"image_size": [800, 600],
		"device": "cuda",
        "seed": 1234
	},
	"model": {
        "class": "ai.models.pytorch.functional._retinanet.model.RetinaNet",
        "kwargs": {
            "backbone": "resnet50",
            "pretrained": True,
            "outPlanes": 256,
            "convertToInstanceNorm": False
        }
	},
    "dataset": {
        "class": "ai.models.pytorch.functional.datasets.bboxDataset.BoundingBoxDataset",
        "kwargs": {
            "targetFormat": "xywh"
        }
    },
	"train": {
        "dataLoader": {
            "kwargs": {
                "shuffle": True,
                "batch_size": 32
            }
        },
		"optim": {
			"class": "torch.optim.Adam",
			"kwargs": {
				"lr": 1e-7,
				"weight_decay": 0.0
			}
		},
        "transform": {
            "class": "ai.models.pytorch.functional._util.bboxTransforms.Compose",
            "kwargs": {
                "transforms": [
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.Resize",
                        "kwargs": {
                            "outSize": [800, 600]
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.RandomHorizontalFlip",
                        "kwargs": {
                            "p": 0.5
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.DefaultTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.ColorJitter",
                                "kwargs": {
                                    "brightness": 0.25,
                                    "contrast": 0.25,
                                    "saturation": 0.25,
                                    "hue": 0.01
                                }
                            }
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.DefaultTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.ToTensor"
                            }
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.DefaultTransform",
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
			"class": "ai.models.pytorch.functional._retinanet.loss.FocalLoss",
			"kwargs": {
				"gamma": 2.0,
				"alpha": 0.25,
				"background_weight": 0.25
			}
		},
        "ignore_unsure": True
	},
	"inference": {
        "transform": {
            "class": "ai.models.pytorch.functional._util.bboxTransforms.Compose",
            "kwargs": {
                "transforms": [
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.Resize",
                        "kwargs": {
                            "outSize": [800, 600]
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.DefaultTransform",
                        "kwargs": {
                            "transform": {
                                "class": "torchvision.transforms.ToTensor"
                            }
                        }
                    },
                    {
                        "class": "ai.models.pytorch.functional._util.bboxTransforms.DefaultTransform",
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
		"batch_size": 256
	}
}