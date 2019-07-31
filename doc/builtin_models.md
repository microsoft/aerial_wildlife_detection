# Configure the built-ins

AIde comes with a number of built-in models for both the AI model as well as the ranker. These are, to a certain extent, customizable, the procedure of which will be explained below.
If you instead wish to completely write your own module(s) for either (or both) of the tasks, you can do so by referring to the instructions [here](custom_model.md).



## AI models

AIde ships with the following AI models built-in:
* Classification:
  * ResNet (He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016), including ResNet-18, 34, 50, 101 and 152.
* Object detection:
  * RetinaNet (Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017), based on the [implementation by Kuangliu](https://github.com/kuangliu/pytorch-retinanet).

All models are implemented using [PyTorch](https://pytorch.org/) and support a number of custom configuration parameters.


To use and configure one of the built-in AI models, you may proceed as follows:
1. Create a JSON file with your custom settings for the model. The default settings per model are outlined below.
2. Provide the correct details in the [configuration *.ini file](configure_settings.md).
For example, to use ResNet for image classification:
```ini
[AIController]

model_lib_path = ai.models.pytorch.ResNet
model_options_path = /path/to/your/settings.json
```



### Default model settings

Below follow the default parameters for each of the built-in models. You may use these as a template for your JSON file to override parameters.

Notes:
* Not all of the arguments specified in the defaults are required. If an argument (or argument block) is missing in your custom JSON file, it will be replaced with the defaults (printed above) instead.
* Any entry named `kwargs` accepts all arguments of the respective Python class (given by the `class` specifier in the same JSON block). For example, you could add the keyword `eps` to the `kwargs` entry of the `optim` section, since this is a valid parameter for e.g. the [Adam optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
* You can further provide custom class executable paths. For example, to add a custom transform function:
    1. Create a new package on the Python path, such as `custom.transforms`.
    2. In your package, add a file with a certain name (e.g. `classificationTransforms.py`). In there, write down your custom class definition (e.g. `MyCustomTransform`).
    3. Link to your custom class by adding the following block to the JSON entry at the right position:
    ```json
    {
        "class": "custom.transforms.classificationTransforms.MyCustomTransform",
        "kwargs": {}
    }
    ```
    Again, `kwargs` is optional, but may contain key-value pairs of class constructor arguments for your custom class.




#### Classification

##### ResNet

```json
{
	"general": {
		"image_size": [224, 224],
		"device": "cuda",
        "seed": 0
	},
	"model": {
        "class": "ai.models.pytorch.ResNet",
        "kwargs": {
			"featureExtractor": "resnet50",
			"pretrained": true
		}
	},
    "dataset": {
		"class": "ai.models.pytorch.ClassificationDataset"
	},
	"train": {
        "dataLoader": {
            "kwargs": {
                "shuffle": true,
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
        "ignore_unsure": true
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
```


#### Detection

##### RetinaNet

```json
{
	"general": {
		"image_size": [800, 600],
		"device": "cuda",
		"seed": 1234
	},
	"model": {
		"kwargs": {
			"backbone": "resnet50",
			"pretrained": false,
			"out_planes": 256,
			"convertToInstanceNorm": false
		}
	},
	"train": {
		"dataLoader": {
			"kwargs": {
				"shuffle": true,
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
			"class": "ai.models.pytorch.detection.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.detection.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.detection.RandomHorizontalFlip",
						"kwargs": {
							"p": 0.5
						}
					},
					{
						"class": "ai.models.pytorch.detection.DefaultTransform",
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
						"class": "ai.models.pytorch.detection.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.detection.DefaultTransform",
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
		"ignore_unsure": true
	},
	"inference": {
		"transform": {
			"class": "ai.models.pytorch.detection.Compose",
			"kwargs": {
				"transforms": [{
						"class": "ai.models.pytorch.detection.Resize",
						"kwargs": {
							"size": [800, 600]
						}
					},
					{
						"class": "ai.models.pytorch.detection.DefaultTransform",
						"kwargs": {
							"transform": {
								"class": "torchvision.transforms.ToTensor"
							}
						}
					},
					{
						"class": "ai.models.pytorch.detection.DefaultTransform",
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
				"shuffle": false,
				"batch_size": 32
			}
		}
	}
}
```


### About object detection transforms

For object detection, certain transforms naturally should not be carried out on the image alone. For example, random horizontal flips affect both the image and the labeled bounding boxes.
AIde therefore treats object detection transforms differently by always applying them to the image, bounding boxes and labels in union. This means that object detection models, such as `RetinaNet`, only accept one of the following transforms as a top-level transform object:

* `ai.models.pytorch.detection.Compose`
  Accepts an iterable of custom, detection-ready transforms and applies them in order.

  Args:
  * transforms (iterable): Iterable (list, tuple, etc.) of transforms from this list here.


* `ai.models.pytorch.detection.DefaultTransform`
  Receives one of the standard PyTorch transforms (e.g. from `torchvision.transforms`) and, at runtime, applies it to the image or tensor only (not to bounding boxes or labels). This is useful for image manipulations that do not result in a geometric change, such as color distortions.

  Args:
  * transform: a callable object that works on either PIL images or torch tensors, depending on where in the transform stack the 'DefaultTransform' is inserted.


* `ai.models.pytorch.detection.Resize`
  Resize an image and associated bounding boxes to a given size.

  Args:
  * size (sequence or int): int (square side) or iterable of size 2 (width, height) denoting the image's absolute target size.
  * interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``.


* `ai.models.pytorch.detection.RandomHorizontalFlip`
  Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

  Args:
  * p (float): probability of the image and boxes being flipped. Default value is 0.5


* `ai.models.pytorch.detection.RandomFlip`
  Horizontally and/or vertically flip the given PIL Image and bounding boxes randomly with a given probability.

  Args:
  * p_h (float): probability of the image and boxes being flipped horizontally. Default value is 0.5
  * p_v (float): probability of the image and boxes being flipped vertically. Default value is 0.5


* `ai.models.pytorch.detection.RandomClip`
  Random image clip of fixed size with custom properties.

  Args:
  * patchSize (sequence or int): int (square side) or iterable of size 2 (width, height) denoting the size of the patch to be clipped from the full image
  * jitter (sequence, int or float): int, float (both directions) or iterable (x, y) denoting the maximum pixel values that are randomly added or subtracted to the X and Y coordinates of the patch center. Useful to maximize variability.
  * limitBorders (boolean): boolean. If True, patches are always fully inside the parent image and never exceed its boundaries.
  * objectProbability (float): either a scalar in [0, 1], or None.
                       If a scalar value is chosen, the patch will be clipped with position from one of the label bounding boxes (if available) at random, under the condition that a uniform, random value is <= this scalar.
                       If set to None, the patch will be clipped completely at random.


* `ai.models.pytorch.detection.RandomSizedClip`
  Random image clip with custom size and custom properties. Similar to `ai.models.pytorch.detection.RandomClip`, but with two additional parameters.
  
  Args:
  * patchSizeMin (sequence or int): int (square side) or iterable of size 2 (width, height) denoting the minimum patch size to be clipped
  * patchSizeMax (sequence or int): int (square side) or iterable of size 2 (width, height) denoting the maximum patch size to be clipped
  * jitter (sequence, int or float)
  * limitBorders (boolean)
  * objectProbability (float)