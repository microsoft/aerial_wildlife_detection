'''
    Default configuration properties for PyTorch classification models.
    May be overridden (also partially) by models subclassig the classification trainer,
    or else through the custom configuration loaded at runtime.

    2019 Benjamin Kellenberger
'''

DEFAULT_OPTIONS = {
	"general": {
		"image_size": [224, 224],
		"device": "cuda",
        "seed": 0
	},
	"model": {
        "class": "ai.models.pytorch.ResNet",
        "kwargs": {
			"featureExtractor": "resnet50",
			"pretrained": True
		}
	},
    "dataset": {
		"class": "ai.models.pytorch.LabelsDataset"
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
			"class": "torchvision.transforms.Compose",
			"kwargs": {
				"transforms": [{
						"class": "torchvision.transforms.Resize",
						"kwargs": {
							"size": [224, 224]
						}
					},
					{
						"class": "torchvision.transforms.RandomHorizontalFlip",
						"kwargs": {
							"p": 0.5
						}
					},
					{
                        "class": "torchvision.transforms.ToTensor"
                    },
                    {
                        "class": "torchvision.transforms.Normalize",
                        "kwargs": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]
                        }
                    }
				]
			}
		},
        "criterion": {
			"class": "torch.nn.CrossEntropyLoss"
		},
        "ignore_unsure": True
	},
	"inference": {
		"transform": {
			"class": "torchvision.transforms.Compose",
			"kwargs": {
				"transforms": [{
                        "class": "torchvision.transforms.Resize",
                        "kwargs": {
                            "size": [224, 224]
                        }
                    },
                    {
                        "class": "torchvision.transforms.ToTensor"
                    },
                    {
                        "class": "torchvision.transforms.Normalize",
                        "kwargs": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]
                        }
                    }
				]
			}
		},
		"batch_size": 256
	}
}