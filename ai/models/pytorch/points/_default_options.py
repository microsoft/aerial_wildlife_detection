'''
    Default configuration properties for PyTorch point prediction models.
    May be overridden (also partially) by models subclassig the classification trainer,
    or else through the custom configuration loaded at runtime.

    2019 Benjamin Kellenberger
'''

DEFAULT_OPTIONS = {
	"general": {
		"image_size": [800, 600],
		"device": "cuda",
        "seed": 0
	},
	"model": {
        "kwargs": {
			"featureExtractor": "resnet50",
			"pretrained": True
		}
	},
    "dataset": {
		"class": "ai.models.pytorch.PointsDataset",
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
			"class": "torch.optim.Adam",
			"kwargs": {
				"lr": 1e-7,
				"weight_decay": 0.0
			}
		},
        "transform": {
			"class": "ai.models.pytorch.points.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.points.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.points.RandomHorizontalFlip",
						"kwargs": {
							"p": 0.5
						}
					},
					{
						"class": "ai.models.pytorch.points.DefaultTransform",
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
						"class": "ai.models.pytorch.points.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.points.DefaultTransform",
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
			"class": "ai.models.pytorch.functional._wsodPoints.loss.PointsLoss",
			"kwargs": {
				"background_weight": 1.0
			}
		},
        "ignore_unsure": True
	},
	"inference": {
		"transform": {
			"class": "ai.models.pytorch.points.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.points.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.points.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.points.DefaultTransform",
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