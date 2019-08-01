'''
    PyTorch implementation of the RetinaNet object detector:
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.

    Basic implementation forked and adapted from: https://github.com/kuangliu/pytorch-retinanet

    2019 Benjamin Kellenberger
'''


# default options for the model, may be overridden in the custom configuration loaded at runtime
DEFAULT_OPTIONS = {
	"general": {
		"image_size": [800, 600],
		"device": "cuda",
		"seed": 1234
	},
	"model": {
		"kwargs": {
			"backbone": "resnet50",
			"pretrained": False,
			"out_planes": 256,
			"convertToInstanceNorm": False
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
			"class": "ai.models.pytorch.boundingBoxes.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.boundingBoxes.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.boundingBoxes.RandomHorizontalFlip",
						"kwargs": {
							"p": 0.5
						}
					},
					{
						"class": "ai.models.pytorch.boundingBoxes.DefaultTransform",
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
						"class": "ai.models.pytorch.boundingBoxes.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.boundingBoxes.DefaultTransform",
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
				"background_weight": 1.0
			}
		},
		"ignore_unsure": True
	},
	"inference": {
		"transform": {
			"class": "ai.models.pytorch.boundingBoxes.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.boundingBoxes.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.boundingBoxes.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.boundingBoxes.DefaultTransform",
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
				"batch_size": 32
			}
		}
	}
}