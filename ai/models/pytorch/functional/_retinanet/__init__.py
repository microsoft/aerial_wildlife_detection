'''
    PyTorch implementation of the RetinaNet object detector:
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.

    Basic implementation forked and adapted from: https://github.com/kuangliu/pytorch-retinanet

    2019 Benjamin Kellenberger
'''


# default options for the model, may be overridden in the custom configuration loaded at runtime
DEFAULT_OPTIONS = {
	"defs": {
		"device": {
			"cpu": {
				"name": "CPU",
				"description": "Run RetinaNet on the CPU and with system RAM."
			},
			"cuda": {
				"name": "GPU",
				"description": "Requires a <a href=\"https://developer.nvidia.com/cuda-zone\" target=\"_blank\">CUDA-enabled</a> graphics card."
			}
		},
		"backbone": {
			"resnet18": {
				"name": "ResNet-18"
			},
			"resnet34": {
				"name": "ResNet-34"
			},
			"resnet50": {
				"name": "ResNet-50"
			},
			"resnet101": {
				"name": "ResNet-101"
			},
			"resnet152": {
				"name": "ResNet-152"
			}
		},
		"transform": {
			"torchvision.transforms.Normalize": {
				"name": "Normalize",
				"description": "Image normalization by band-wise mean subtraction and standard deviation division.<br />Default values are optimized for models pre-trained on ImageNet.",
				"mean": {
					"name": "Mean values",
					"type": "list",
					"value": [{
							"name": "Red",
							"min": 0,
							"max": 1e9,
							"value": 0.485
						},
						{
							"name": "Green",
							"min": 0,
							"max": 1e9,
							"value": 0.456
						},
						{
							"name": "Blue",
							"min": 0,
							"max": 1e9,
							"value": 0.406
						}
					]
				},
				"std": {
					"name": "Standard deviation values",
					"type": "list",
					"value": [{
							"name": "Red",
							"min": 0,
							"max": 1e9,
							"value": 0.229
						},
						{
							"name": "Green",
							"min": 0,
							"max": 1e9,
							"value": 0.224
						},
						{
							"name": "Blue",
							"min": 0,
							"max": 1e9,
							"value": 0.225
						}
					]
				}
			},
			"torchvision.transforms.ColorJitter": {
				"name": "Color Jitter",
				"description": "Randomly changes the images' brightness, contrast, hue and saturation",
				"brightness": {
					"name": "Brightness",
					"type": "list",
					"value": [{
							"name": "Min",
							"min": 0,
							"max": 1e9,
							"value": 0.0,
							"style": {
								"inline": True
							}
						},
						{
							"name": "Max",
							"min": 0,
							"max": 1e9,
							"value": 0.25,
							"style": {
								"inline": True
							}
						}
					]
				},
				"contrast": {
					"name": "Contrast",
					"type": "list",
					"value": [{
							"name": "Min",
							"min": 0,
							"max": 1e9,
							"value": 0.0,
							"style": {
								"inline": True
							}
						},
						{
							"name": "Max",
							"min": 0,
							"max": 1e9,
							"value": 0.25,
							"style": {
								"inline": True
							}
						}
					]
				},
				"saturation": {
					"name": "Saturation",
					"type": "list",
					"value": [{
							"name": "Min",
							"min": 0,
							"max": 1e9,
							"value": 0.0,
							"style": {
								"inline": True
							}
						},
						{
							"name": "Max",
							"min": 0,
							"max": 1e9,
							"value": 0.25,
							"style": {
								"inline": True
							}
						}
					]
				},
				"hue": {
					"name": "Hue",
					"type": "list",
					"value": [{
							"name": "Min",
							"min": 0,
							"max": 1e9,
							"value": 0.0,
							"style": {
								"inline": True
							}
						},
						{
							"name": "Max",
							"min": 0,
							"max": 1e9,
							"value": 0.01,
							"style": {
								"inline": True
							}
						}
					]
				}
			},
			"torchvision.transforms.Grayscale": {
				"name": "Grayscale",
				"description": "Convert image to grayscale"
			},
			"ai.models.pytorch.boundingBoxes.RandomHorizontalFlip": {
				"name": "Random horizontal flip",
				"description": "Randomly flips the image along the horizontal axis with a probability",
				"p": {
					"name": "Probability",
					"min": 0,
					"max": 1.0,
					"value": 0.5,
					"style": {
						"inline": True,
						"slider": True
					}
				}
			},
			"ai.models.pytorch.boundingBoxes.RandomFlip": {
				"name": "Random flip",
				"description": "Randomly flips the image along both horizontal and vertical axes with a probability",
				"p_h": {
					"name": "Probability (horizontal)",
					"min": 0,
					"max": 1.0,
					"value": 0.5,
					"style": {
						"inline": True,
						"slider": True
					}
				},
				"p_v": {
					"name": "Probability (vertical)",
					"min": 0,
					"max": 1.0,
					"value": 0.5,
					"style": {
						"inline": True,
						"slider": True
					}
				}
			},
			"ai.models.pytorch.boundingBoxes.RandomClip": {
				"name": "Random clip",
				"description": "Clip a patch of predefined size from a (semi-) random location in the image",
				"patchSize": {
					"name": "Patch size",
					"description": "Width and height of the patch to clip",
					"width": {
						"name": "Width",
						"min": 1,
						"max": 1e9,
						"style": {
							"inline": True
						}
					},
					"height": {
						"name": "Height",
						"min": 1,
						"max": 1e9,
						"style": {
							"inline": True
						}
					}

				},
				"jitter": {
					"name": "Jitter",
					"description": "How much noise (in pixels) to add when clipping around bounding boxes",
					"x": {
						"name": "X",
						"description": "Maximum jitter amount in horizontal direction [pixels]",
						"min": 0,
						"max": 1e9,
						"value": 0,
						"style": {
							"inline": True
						}
					},
					"y": {
						"name": "Y",
						"description": "Maximum jitter amount in vertical direction [pixels]",
						"min": 0,
						"max": 1e9,
						"value": 0,
						"style": {
							"inline": True
						}
					}

				},
				"limitBorders": {
					"name": "Limit patch to image borders",
					"value": True
				},
				"objectProbability": {
					"name": "Object probability",
					"description": "Probability to center patch around one of the bounding boxes in an image (else random location)",
					"min": 0,
					"max": 1,
					"value": 0.5,
					"style": {
						"inline": True,
						"slider": True
					}
				}
			},
			"ai.models.pytorch.boundingBoxes.RandomSizedClip": {
				"name": "Random sized clip",
				"description": "Clip a patch of random size from a (semi-) random location in the image",
				"patchSize": {
					"name": "Patch sizes range",
					"description": "Width and height of the patch to clip",
					"width_min": {
						"name": "Minimum width",
						"min": 1,
						"max": 1e9,
						"value": 400,
						"style": {
							"inline": True
						}
					},
					"width_max": {
						"name": "Maximum width",
						"min": 1,
						"max": 1e9,
						"value": 800,
						"style": {
							"inline": True
						}
					},
					"height_min": {
						"name": "Minimum height",
						"min": 1,
						"max": 1e9,
						"value": 300,
						"style": {
							"inline": True
						}
					},
					"height_max": {
						"name": "Maximum height",
						"min": 1,
						"max": 1e9,
						"value": 600,
						"style": {
							"inline": True
						}
					}
				},
				"jitter": {
					"name": "Jitter",
					"description": "How much noise in spatial position to add when clipping around bounding boxes",
					"x": {
						"name": "X",
						"description": "Jitter amount in horizontal direction",
						"min": 0,
						"max": 1e9,
						"value": 0,
						"style": {
							"inline": True
						}
					},
					"y": {
						"name": "Y",
						"description": "Jitter amount in vertical direction",
						"min": 0,
						"max": 1e9,
						"value": 0,
						"style": {
							"inline": True
						}
					}
				},
				"limitBorders": {
					"name": "Limit patch to image borders",
					"value": True
				},
				"objectProbability": {
					"name": "Object probability",
					"description": "Probability to center patch around one of the bounding boxes in an image (else random location)",
					"min": 0,
					"max": 1,
					"value": 0.5,
					"style": {
						"inline": True,
						"slider": True
					}
				}
			}
		},
		"transform_inference": [
			"torchvision.transforms.Normalize",
			"torchvision.transforms.ColorJitter",
			"torchvision.transforms.Grayscale"
		]
	},
	"options": {
		"general": {
			"name": "General options",
			"device": {
				"name": "Device",
				"type": "select",
				"options": "device",
				"value": "cuda",
				"style": {
					"inline": True
				}
			},
			"seed": {
				"name": "Random seed",
				"type": "int",
				"min": 0,
				"max": 1e9,
				"value": 0
			},
			"imageSize": {
				"name": "Image size",
				"description": "Images have to be resized to the same dimensions for training and inference",
				"width": {
					"name": "Width",
					"type": "int",
					"min": 1,
					"max": 50000,
					"value": 800
				},
				"height": {
					"name": "Height",
					"type": "int",
					"min": 1,
					"max": 50000,
					"value": 600
				},
				"interpolation": {
					"name": "Interpolation method",
					"type": "select",
					"options": {
						"Image.NEAREST": {
							"name": "Nearest neighbor"
						},
						"Image.BILINEAR": {
							"name": "Bilinear"
						},
						"Image.BICUBIC": {
							"name": "Bicubic"
						},
						"Image.LANCZOS": {
							"name": "Lanczos"
						}
					},
					"value": "Image.BILINEAR"
				}
			},
			"labelClasses": {
				"name": "New and removed label classes",
				"add_missing": {
					"name": "Add new label classes",
					"description": "If checked, neurons for newly added label classes will be added to the model.<br />Note that these new classes need extra training.",
					"value": True
				},
				"remove_obsolete": {
					"name": "Remove obsolete label classes",
					"description": "If checked, neurons from label classes not present in this project will be removed during next model training.",
					"value": False
				}
			}
		},
		"model": {
			"name": "Model options",
			"backbone": {
				"name": "Model backbone",
				"description": "<a href=\"http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf\" target=\"_blank\">ResNet</a>-based feature extractor to use. Larger models (e.g., ResNet-152) provide higher learning capacity, but require more data to train.",
				"type": "select",
				"options": "backbone",
				"value": "resnet18",
				"style": {
					"inline": True
				}
			},
			"pretrained": {
				"name": "Model weights pre-trained on ImageNet",
				"description": "If checked, model backbone parameters will be pre-trained on the <a href=\"http://image-net.org/\" target=\"_blank\">ImageNet</a> classification challenge for the first training epoch.",
				"value": True
			},
			"out_planes": {
				"name": "Number of intermediate filters",
				"description": "This is the number of filters used in the lateral connections in RetinaNet's <a href=\"http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf\" target=\"_blank\">Feature Pyramid Network</a>. Higher numbers (e.g., 256) may result in higher model capacity, but take longer and require more data to train.",
				"type": "int",
				"min": 0,
				"max": 1024,
				"value": 256
			},
			"convertToInstanceNorm": {
				"name": "Convert Batch normalization to <a href=\"https://arxiv.org/pdf/1607.08022.pdf\" target=\"_blank\">instance normalization</a> layers",
				"value": False
			}
		},
		"train": {
			"name": "Training options",
			"dataLoader": {
				"name": "Data loader options",
				"shuffle": {
					"name": "Shuffle image order",
					"value": True
				},
				"batch_size": {
					"name": "Batch size",
					"description": "Number of images to train on at a time (in chunks). Reduce number in case of out-of-memory issues.",
					"min": 1,
					"max": 8192,
					"value": 1
				}
			},
			"optim": {
				"name": "Optimizer",
				"type": "select",
				"options": {
					"torch.optim.Adadelta": {
						"name": "Adadelta",
						"description": "<a href=\"https://arxiv.org/abs/1212.5701\" target=\"_blank\">Adadelta optimizer</a>",
						"lr": {
							"name": "Learning rate",
							"min": 0.0,
							"max": 100.0,
							"value": 1.0
						},
						"weight_decay": {
							"name": "Weight decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"rho": {
							"name": "Rho",
							"description": "Coefficient for running average of squared gradients",
							"min": 0.0,
							"max": 100.0,
							"value": 0.9
						}

					},
					"torch.optim.Adagrad": {
						"name": "Adagrad",
						"description": "<a href=\"http://jmlr.org/papers/v12/duchi11a.html\" target=\"_blank\">Adadelta optimizer</a>",
						"lr": {
							"name": "Learning rate",
							"min": 0.0,
							"max": 100.0,
							"value": 1e-2
						},
						"lr_decay": {
							"name": "Learning rate decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"weight_decay": {
							"name": "Weight decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						}

					},
					"torch.optim.Adam": {
						"name": "Adam",
						"description": "<a href=\"https://arxiv.org/pdf/1412.6980.pdf\" target=\"_blank\">Adam optimizer</a>",
						"lr": {
							"name": "Learning rate",
							"min": 0.0,
							"max": 100.0,
							"value": 1e-4
						},
						"weight_decay": {
							"name": "Weight decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						}
					},
					"torch.optim.RMSprop": {
						"name": "RMSprop",
						"description": "<a href=\"https://arxiv.org/pdf/1308.0850v5.pdf\" target=\"_blank\">RMSprop optimizer</a>",
						"lr": {
							"name": "Learning rate",
							"min": 0.0,
							"max": 100.0,
							"value": 1e-2
						},
						"weight_decay": {
							"name": "Weight decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"momentum": {
							"name": "Momentum",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"alpha": {
							"name": "Alpha",
							"description": "Smoothing constant",
							"min": 0.0,
							"max": 100.0,
							"value": 0.99
						},
						"centered": {
							"name": "Normalize gradient by estimated variance (\"centered RMSprop\")",
							"value": False
						}
					},
					"torch.optim.SGD": {
						"name": "Stochastic Gradient Descent",
						"description": "Stochastic gradient descent optimizer",
						"lr": {
							"name": "Learning rate",
							"min": 0.0,
							"max": 100.0,
							"value": 1e-4
						},
						"weight_decay": {
							"name": "Weight decay",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"momentum": {
							"name": "Momentum",
							"min": 0.0,
							"max": 100.0,
							"value": 0.9
						},
						"dampening": {
							"name": "Dampening for momentum",
							"min": 0.0,
							"max": 100.0,
							"value": 0.0
						},
						"nesterov": {
							"name": "Enable <a href=\"http://www.cs.toronto.edu/~hinton/absps/momentum.pdf\" target=\"_blank\">Nesterov momentum</a>",
							"value": False
						}
					}
				},
				"value": "torch.optim.SGD",
				"style": {
					"inline": True
				}
			},
			"transform": {
				"name": "Transforms",
				"description": "Transforms are used to prepare images as inputs for the model, as well as to perform data augmentation.",
				"type": "list",
				"options": "transform",
				"value": [
					"ai.models.pytorch.boundingBoxes.RandomHorizontalFlip",
					"torchvision.transforms.ColorJitter",
					"torchvision.transforms.Normalize"
				]
			},
			"criterion": {
				"name": "Focal Loss options",
				"description": "See the <a href=\"http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf\" target=\"_blank\">RetinaNet paper</a> for more information.",
				"gamma": {
					"name": "Gamma",
					"min": 0.0,
					"max": 8192,
					"value": 2.0
				},
				"alpha": {
					"name": "Alpha",
					"min": 0.0,
					"max": 8192,
					"value": 0.25
				},
				"background_weight": {
					"name": "Background weight",
					"min": 0.0,
					"max": 8192,
					"value": 1.0
				}
			},
			"encoding": {
				"name": "Bounding box encoding",
				"minIoU_pos": {
					"name": "Min. IoU for positives",
					"description": "Minimum IoU value between a target and prediction box for the latter to be considered a correct prediction",
					"min": 0.0,
					"max": 1.0,
					"value": 0.5,
					"style": {
						"slider": True
					}
				},
				"maxIoU_neg": {
					"name": "Max. IoU for negatives",
					"description": "Maximally permitted IoU value between a target and prediction box for the latter to be treated as a False positive",
					"min": 0.0,
					"max": 1.0,
					"value": 0.4,
					"style": {
						"slider": True
					}
				},
				"ignore_unsure": {
					"name": "Ignore annotations marked as \"unsure\"",
					"value": True
				}
			}
		},
		"inference": {
			"name": "Inference (prediction) options",
			"dataLoader": {
				"name": "Data loader options",
				"batch_size": {
					"name": "Batch size",
					"description": "Number of images to predict objects in at a time. Reduce number in case of out-of-memory issues.",
					"min": 1,
					"max": 8192,
					"value": 1
				}
			},
			"transform": {
				"name": "Transforms",
				"description": "Note that inference transforms exclude geometric data augmentation options.",
				"type": "list",
				"options": "transform_inference",
				"value": [
					"torchvision.transforms.Normalize"
				]
			},
			"encoding": {
				"name": "Bounding box encoding",
				"cls_thresh": {
					"name": "Min. class confidence",
					"description": "Minimum confidence value to be reached by any label class for a box to be kept as a prediction.<br />Higher values = more confident predictions; lower values = higher recall",
					"min": 0.0,
					"max": 1.0,
					"value": 0.1,
					"style": {
						"slider": True
					}
				},
				"nms_thresh": {
					"name": "Non-maximum suppression threshold",
					"description": "Maximally permitted IoU between predictions. If above, boxes with lower confidence scores will be discarded (non-maximum suppression).",
					"min": 0.0,
					"max": 1.0,
					"value": 0.1,
					"style": {
						"slider": True
					}
				},
				"numPred_max": {
					"name": "Maximum number of predictions",
					"description": "Limit the number of predicted boxes to the value given. This is especially useful at the beginning of model training, where a high number of boxes is usually predicted.",
					"type": "int",
					"min": 1,
					"max": 1000,
					"value": 128,
					"style": {
						"slider": True
					}
				}
			}
		}
	}
}
