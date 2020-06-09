'''
    TensorFlow implementation of the YOLO v3 object detector:
        Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).

    Implementation adapted from: https://github.com/experiencor/keras-yolo3

    2019 Benjamin Kellenberger
    2020 Colin Torney
'''


# default options for the model, may be overridden in the custom configuration loaded at runtime
# - use pretrained=True to load weights file trained on COCO dataset. Weights file can be downloaded from
#   here https://www.dropbox.com/s/bowtcxu117zt6nt/yolo-v3-coco.h5?dl=0 and should be in directory weights
# - use alltrain to train the whole network not just the top layers
# - specify init_weights if initial training outside of AIDE has been done
DEFAULT_OPTIONS = {
	"general": {
		"device": "cuda",
		"seed": 1234
	},
	"model": {
		"kwargs": {
			"pretrained": True,
			"alltrain": False,
                        "init_weights": "weights/trained-camera-trap-yolo.h5"
		}
	},
	"train": {
        "width": 864,
        "height": 864,
		"dataLoader": {
			"kwargs": {
				"shuffle": True,
				"batch_size": 1
			}
		},
		"optim": {
			"class": "tensorflow.keras.optimizers.Adam",
			"kwargs": {
				"learning_rate": 1e-7
			}
		},
		"transform": {
			"class": "ai.models.tensorflow.yolo.boundingBoxes.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.tensorflow.yolo.boundingBoxes.Resize",
						"kwargs": {
							"size": [864, 864]
						}
					},
					{
						"class": "ai.models.tensorflow.yolo.boundingBoxes.RandomHorizontalFlip",
						"kwargs": {
							"p": 0.5
						}
					}
                ]
			}
		},
		"criterion": {
			"class": "ai.models.tensorflow.functional._yolo_3.loss.YoloLoss",
			"kwargs": {
                "NO_OBJECT_SCALE": 1.0,
                "OBJECT_SCALE": 5.0,
                "COORD_SCALE": 4.0,
                "CLASS_SCALE": 2.0
			}
		},
		"ignore_unsure": True
	},
	"inference": {
        "shuffle": False,
        "batch_size": 1,
        "nms_thresh": 0.1,
        "cls_thresh": 0.1,
        	"transform": {
			"class": "ai.models.tensorflow.yolo.boundingBoxes.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.tensorflow.yolo.boundingBoxes.Resize",
						"kwargs": {
							"size": [1952, 2592]
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
