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
		"dataType": "featureVector",		# one of {'image','featureVector'}
        "seed": 1234
	},
	"model": {
		"backbone": "resnet50",
		"pretrained": True,
		"outPlanes": 256,
		"convertToInstanceNorm": False
	},
	"train": {
		"optim": {
			"class": "torch.optim.Adam",
			"kwargs": {
				"lr": 1e-6,
				"weight_decay": 0.0
			}
		},
        "criterion": {
			"class": "ai.models.pytorch.functional._retinanet.loss.FocalLoss",
			"kwargs": {
				"gamma": 2.0,
				"alpha": 0.25,
				"background_weight": 0.5
			}
		},
		"batch_size": 32,
		"ignore_unsure": True
	},
	"inference": {
		"batch_size": 256
	}
}