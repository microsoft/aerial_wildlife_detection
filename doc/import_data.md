# Import existing data into AIde

AIde ships with a couple of helper scripts that allow importing existing datasets into the platform. All of the scripts (and instructions below) need to be executed from the _FileServer_ instance. This means that you must have [installed](install.md) and [configured](configure_settings.md) AIde on that machine accordingly, and to have the [database instance running](setup_db.md).


## Import images only

TODO


## Import an object detection dataset

We provide a script that automatically parses and imports an object detection dataset (i.e., containing images and bounding boxes) into AIde.

### Requirements
This procedure requires the dataset to be organized as follows:

* One folder contains all images (may or may not be hierarchical, with sub-folders)
* A second folder contains annotations as follows:
    - There must be one annotation text file for every image with at least one object in it
    - Annotation text files must be organized in exactly the same way as the images w.r.t. folder hierarchies
    - The only permitted differences between the annotation files' and the images' file names are the source folder and the file extension
    - Annotation text files must list the bounding boxes in [YOLO](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) format (see below)
    - The annotations folder must further contain a "classes.txt" file that lists the class names in the dataset (one human-friendly class name per line)


Example for a valid dataset structure:
```
    images/
        A/
            img_0001.jpg
            img_0002.jpg
        B/
            sub_1/
                DSC_1232.JPEG
                DSC_4433.JPG
            sub_2/
                IMG_3311.png

    labels/
        classes.txt
        A/
            img_0001.txt
        B/
            sub_1/
                DSC_1232.txt
                DSC_4433.txt
```

Note the following:
* Not every image has an associated annotation file under `labels/`. This is being interpreted as "the image contains no object."
* The images' folder structure is exactly replicated for the labels.
* File extensions may vary.
* The `labels/` directory must always contain a `classes.txt` file, with contents explained below.


Annotation text files may be organized as follows:
* One line per annotation
* Tokens in the line are whitespace-separated and in order: `class index`, `x`, `y`, `width`, `height`
  Additionally, we also allow confidence scores to be appended after the last token, which is required if the annotations are to be treated as predictions made by a model.
* Bounding box _position_ coordinates (`x`, `y`) denote the center of the rectangle
* All bounding box values are scaled to the `[0, 1]` range as fractions of the image's width, resp. height.

Example for a valid annotation file:
```
    0 0.33 0.02 0.01 0.3
    3 0.75 0.6643 0.2231355422 0.23
```

Example for a valid annotation file that also contains model predictions:
```
    1 0.07268325239419937 0.33578112721443176 0.024743778631091118 0.06437072902917862 0.3286268711090088 0.9998925924301147
    1 0.13860762119293213 0.018792062997817993 0.03474283963441849 0.032541967928409576 1.0940264319470039e-11 0.9992766976356506
    0 0.11322460323572159 0.43084415793418884 0.020322661846876144 0.03316088765859604 0.9980218410491943 0.001389735727570951
    1 0.15672338008880615 0.8822341561317444 0.03858977183699608 0.031167175620794296 0.010252226144075394 0.9652772545814514
    1 0.8120267391204834 0.31740111112594604 0.022901205345988274 0.06216927990317345 5.7355006161108335e-12 0.9556573033332825
```

Notes:
* The first value is the class index as referenced in the `classes.txt` file (see below). This _must_ be an integer.
* All other values must be floats in `[0, 1]`; scientific notation and varying number of decimal points are allowed.
* In the second example, the confidence scores for each bounding box are appended after the `height` value of the box. Rule: every class in the dataset must be present in the order defined in the `classes.txt` file.


The `classes.txt` file simply lists all label classes in order. The individual class index, which the first number in each annotation file row refers to, is given implicitly by the line number of the file.
Example for a valid `classes.txt` file:
```
    Elephant
    Giraffe
    Human
    Bird
```
In this case, every annotation (or prediction) with class label zero is treated as an `Elephant`, one is a `Giraffe`, etc.



### Importing the data

1. Copy all the _contents_ of the `images/` folder into the `staticfiles_dir` of the _FileServer_ instance, as defined in the [configuration file](configure_settings.md). This copies the images over to the file server; the script run in the steps below will then add them to the database accordingly.


2. Once finished, run the import script from the AIde root with environment variables set on the _FileServer_ instance:
```
    python projectCreation/import_YOLO_dataset.py --label_folder=labels
```
Replace `labels` with the path to the labels folder of your dataset.

This will import all image paths from the _FileServer_'s `staticfiles_dir`, all label classes with names as defined in `classes.txt`, and all labels present as type `annotation` into the database. The annotation/prediction columns `timeRequired` and `timeCreated` will be set to the value -1 and the current date, respectively.

Alternatively, the import script accepts the following parameters:
* `annotation_type`: set to `prediction` to import the labels as predictions instead. Default is `annotation`.
* `al_criterion`: the criterion to use to calculate the priority value for each prediction. May be one of the following:
    - `none`: do not calculate any priority value (will be set to `NULL` in the database).
    - `BreakingTies`: calculates the priority value using the [Breaking Ties](http://www.jmlr.org/papers/volume6/luo05a/luo05a.pdf) criterion. Requires class logits to be appended to each row in the label text files.
    - `MaxConfidence`: uses the value of the class predicted with the highest confidence as a priority value. Requires class logits to be appended to each row in the label text files.
    - `TryAll`: uses `max(BreakingTies, MaxConfidence)` as a criterion to calculate the priority value. Requires class logits to be appended to each row in the label text files.