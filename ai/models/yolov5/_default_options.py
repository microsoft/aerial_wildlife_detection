'''
    2022 Benjamin Kellenberger
'''


DEFAULT_OPTIONS = {
	"defs": {
        "models": {     #TODO: add all models from here: https://github.com/ultralytics/yolov5/tree/master/models/hub
            "yolov5x": {
                "name": "YOLOv5 XLarge"
            },
            "yolov5l": {
                "name": "YOLOv5 Large"
            },
            "yolov5m": {
                "name": "YOLOv5 Medium"
            },
            "yolov5s": {
                "name": "YOLOv5 Small"
            },
            "yolov5n": {
                "name": "YOLOv5 Nano"
            }
        }
	},
	"options": {
		"general": {
			"name": "General options",
			"device": {
				"name": "Device",
				"type": "select",
				"options": {
					"cpu": {
						"name": "CPU",
						"description": "Run YOLOv5 on the CPU and with system RAM."
					},
					"cuda": {
						"name": "GPU",
						"description": "Requires a <a href=\"https://developer.nvidia.com/cuda-zone\" target=\"_blank\">CUDA-enabled</a> graphics card."
					}
				},
				"value": "cuda",
				"style": {
					"inline": True
				}
			},
			"seed": {
				"name": "Random seed",
				"type": "int",
				"min": -1e9,
				"max": 1e9,
				"value": -1
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
			"description": "Choose a pre-trained starting point.<br />If your project already contains at least one model state, this choice is ignored unless \"Force new model\" is ticked, in which case a completely new model is being built.",
			"config": {
				"name": "Pre-made configuration",
				"description": "Choose a pre-trained model state here or create your own model from scratch (\"manual\").",
				"type": "select",
				"value": "yolov5s",
				"options": "models"
			},
			"force": {
				"name": "Force new model",
				"value": False
			}
		},
		"train": {
			"name": "Training options",
			"yolov5.hyp.lr0": {
                "name": "Initial learning rate",
                "value": 0.01,
                "min": 0.0
            },
            "yolov5.hyp.lrf": {
                "name": "Final learning rate (OneCycleLR)",
                "value": 0.01,
                "min": 0.0
            },
            "yolov5.hyp.momentum": {
                "name": "Momentum (SGD) / Beta 1 (Adam)",
                "value": 0.937,
                "min": 0.0
            },
            "yolov5.hyp.weight_decay": {
                "name": "Weight decay",
                "value": 0.0005,
                "min": 0.0
            },
            "yolov5.hyp.warmup_epochs": {
                "name": "Warmup epochs",
                "value": 3.0,
                "min": 0.0
            },
            "yolov5.hyp.warmup_momentum": {
                "name": "Warmup momentum",
                "value": 0.8,
                "min": 0.0
            },
            "yolov5.hyp.warmup_bias_lr": {
                "name": "Warmup initial bias learning rate",
                "value": 0.1,
                "min": 0.0
            },
            "yolov5.hyp.box": {
                "name": "Bounding box loss gain",
                "value": 0.05,
                "min": 0.0
            },
            "yolov5.hyp.cls": {
                "name": "Classification loss gain",
                "value": 0.5,
                "min": 0.0
            },
            "yolov5.hyp.cls_pw": {
                "name": "Classification Binary Cross-Entropy loss weight for positives",
                "value": 1.0,
                "min": 0.0
            },
            "yolov5.hyp.obj": {
                "name": "Object loss gain (scales with pixels)",
                "value": 1.0,
                "min": 0.0
            },
            "yolov5.hyp.obj_pw": {
                "name": "Object Binary Cross-Entropy loss weight for positives",
                "value": 1.0,
                "min": 0.0
            },
            "yolov5.hyp.iou_t": {
                "name": "Intersection-over-Union training threshold",
                "value": 0.2,
                "min": 0.0,
                "max": 1.0
            },
            "yolov5.hyp.anchor_t": {
                "name": "Anchor multiple threshold",
                "value": 4.0,
                "min": 0.0
            },
            "yolov5.hyp.fl_gamma": {
                "name": "Focal loss gamma",
                "value": 0.0,
                "min": 0.0
            },
            "transforms": {
                "hsv": {
                    "name": "HSV transform",
                    "yolov5.hyp.hsv_h": {
                        "name": "Hue augmentation fraction",
                        "value": 0.015,
                        "min": 0.0,
                        "max": 1.0
                    },
                    "yolov5.hyp.hsv_s": {
                        "name": "Saturation augmentation fraction",
                        "value": 0.7,
                        "min": 0.0,
                        "max": 1.0
                    },
                    "yolov5.hyp.hsv_v": {
                        "name": "Value (brightness) augmentation fraction",
                        "value": 0.4,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "rot": {
                    "name": "Image rotation",
                    "yolov5.hyp.degrees": {
                        "name": "Maximum degrees of random rotation",
                        "value": 0.0,
                        "min": 0.0,
                        "max": 360.0
                    }
                },
                "tran": {
                    "name": "Image translation",
                    "yolov5.hyp.translate": {
                        "name": "Image translation fraction",
                        "value": 0.1,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "scale": {
                    "name": "Image scaling",
                    "yolov5.hyp.scale": {
                        "name": "Image scaling gain",
                        "value": 0.5,
                        "min": 0.0
                    }
                },
                "shear": {
                    "name": "Image shearing",
                    "yolov5.hyp.shear": {
                        "name": "Image shearing (+/- degrees)",
                        "value": 0.0
                    }
                },
                "perspective": {
                    "name": "Image perspective",
                    "yolov5.hyp.perspective": {
                        "name": "Image perspective (+/- fraction)",
                        "value": 0.0,
                        "min": 0.0,
                        "max": 0.001
                    }
                },
                "flip": {
                    "name": "Image flipping",
                    "yolov5.hyp.flipud": {
                        "name": "Up/down probability",
                        "value": 0.0,
                        "min": 0.0,
                        "max": 1.0
                    },
                    "yolov5.hyp.fliplr": {
                        "name": "Left/right probability",
                        "value": 0.5,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "mosaic": {
                    "name": "Image mosaicking",
                    "yolov5.hyp.mosaic": {
                        "name": "Mosaicking probability",
                        "value": 1.0,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "mixup": {
                    "name": "Image mix-up",
                    "yolov5.hyp.mixup": {
                        "name": "Mix-up probability",
                        "value": 0.0,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "copypaste": {
                    "name": "Segment copy-pasting",
                    "yolov5.hyp.copy_paste": {
                        "name": "Copy-paste probability",
                        "value": 0.0,
                        "min": 0.0,
                        "max": 1.0
                    }
                }
            },
            "batch_size": {
				"name": "Batch size",
				"value": 2,
                "min": 1
			},
            "optim": {
				"name": "Optimizer",
				"type": "select",
				"options": {
					"torch.optim.Adadelta": {
						"name": "Adadelta",
                        "class": "Adadelta",
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
                        "class": "Adagrad",
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
                        "class": "Adam",
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
                        "class": "RMSProp",
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
                        "class": "SGD",
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
			"ignore_unsure": {
				"name": "Ignore (discard) annotations marked as \"unsure\"",
				"value": True
			},
            "filter_empty": {
				"name": "Ignore (discard) images with zero annotations",
				"value": False
			}
		},
		"inference": {
			"name": "Inference (prediction) options",
			
		}
	}
}