# Custom AI Backend

AIDE supports the implementation of arbitrary prediction models, using any framework as long as they are implemented in, or provide the appropriate interface to, Python.
In detail, the following portions are customizable:
* The prediction model itself
* The ranking model (_i.e._, the model providing priority scores for predictions)
* The annotation filtering model (TBA)

Note that some models are already built-in and can be configured, so that you might not need to write a custom model at all. Check out the associated [manual page](builtin_models.md) for details.


Once a custom model is implemented, it can be plugged into a platform instance by providing its Python package path as an argument to the [configuration *.ini file](configure_settings.md).

The following sections will provide details on how to implement custom models for each portion.


## Implement a custom prediction model

Below is a sample code shell for a custom prediction model:

```python

    class MyCustomModel:
        def __init__(self, config, dbConnector, fileServer, options):
            """
                Model constructor. This is called by both the AIWorker and AIController
                modules when starting up.
                Args:
                    config: Configuration for the current AIDE project
                    dbConnector: Access to the project database
                    fileServer: Access to the instance storing the images
                    options: A custom set of options in JSON format for this model
            """
            self.config = config
            self.dbConnector = dbConnector
            self.fileServer = fileServer
            self.options = options


        def train(self, stateDict, data):
            """
                Training function. This function gets called by each individual AIWorkers
                when the model is supposed to be trained for another round.
                Args:
                    stateDict: a bytes object containing the model's current state
                    data: a dict object containing the image metadata to be trained on
                
                Returns:
                    stateDict: a bytes object containing the model's state after training
            """

            raise NotImplementedError('to be implemented')
        

        def average_model_states(self, stateDicts):
            """
                Averaging function. If AIDE is configured to distribute training to multiple
                AIWorkers, and if multiple AIWorkers are attached, this function will be called
                by exactly one AIWorker after the "train" function has finished.
                Args:
                    stateDicts: a list of N bytes objects containing model states as trained by
                                the N AIWorkers attached

                Returns:
                    stateDict: a bytes object containing the combined model states
            """

            raise NotImplementedError('to be implemented')


        def inference(self, stateDict, data):
            """
                Inference function. This gets called everytime the model is supposed to be run on
                a set of images. The inference job might be distributed to multiple AIWorkers, but
                there is no need to worry about concurrency or race conditions, as each inference
                job is handled separately.
                Args:
                    stateDict: a bytes object containing the latest model state
                    data: a dict object containing the metadata of the images the model needs to
                          predict
            """

            raise NotImplementedError('to be implemented')
```

Note the following:
* The layout given in the sample code above must not be changed (i.e., it must
  exactly contain the functions and their arguments specified).
* You may add more custom functions, packages, files, etc. the model can rely on,
  though.


### Data and argument specifications

#### Constructor parameters
* **config:** Provides the model with configuration parameters as specified in the [configuration *.ini file](configure_settings.md). To access a parameter, use the following syntax (example):
```python
    param = config.getProperty('Project', 'annotationType', type=str, fallback=None)
```
This example would return the `annotationType` entry under section `Project`, or `None` if not present. Arguments `type` (default: `type=str`) and `fallback` are optional. If parameter not present and `fallback` not specified, the function raises an exception.

* **dbConnector:** Provides access to the project database. Note: this is only needed under exceptional circumstances. **There is no need to manually store annotations or model states; this is taken care of by the _AIWorker_ directly.**
If you do need to access the database, you can do so as follows (example):
```python
    sql = 'SELECT * FROM {schema}.annotation WHERE x > %s;'.format(schema=config.getProperty('Database', 'schema'))
    arguments = 0.5
    numReturn = 'all'
    result = dbConnector.execute(sql, (arguments,), numReturn)
```
This would return a list of dicts with values from the `annotation` table.

* **fileServer:** This is the most typically used helper, as it returns images needed for training and inference.
Example:
```python
    import io
    from PIL import Image
    filename = '/local/path/IMG_0001.JPG'
    imgBytes = BytesIO(fileServer.getFile(filename))
    image = Image.open(imgBytes)
```
This code requests an image with given path from the _FileServer_ instance and opens it as a [PIL Image](https://pillow.readthedocs.io/en/stable/). You may also use other libraries and formats than PIL, such as [TensorFlow records](https://www.tensorflow.org/tutorials/load_data/images). Note that results are returned as bytes and need to be read and converted.
**Note:** if an image cannot be found, or any other error occurs, `None` is returned.

* **options:** These are parameters specific to the model. Use this for e.g. setting the model's learning rate, batch size, etc. Options can be provided through a JSON file; this requires setting the 'model_options_path' to the file path of the file in the [configuration *.ini file](configure_settings.md).


#### Training parameters
* **stateDict:** This parameter contains a bytes object of the latest model state. Note that the actual contents of the `stateDict` are not restricted explicitly and may be set arbitrarily by the model you design. The only absolute restrictions are:
- The data for the `stateDict` need to be serializable;
- You need to provide an actual bytes array.
So, for example, you can save your model state into a Python dict and use the pickle module to serialize it.
Loading such an object could then look as follows:
```python
    import pickle
    import io
    stateDict = pickle.load(io.BytesIO(stateDict))
```

* **data:** This contains the metadata for all the images the model needs to be trained on. Data come as a Python dict according to the following specifications:
    ```python
        data = {
            'images': {
                '83d7b609-e3d1-45fb-8701-79f50d25087c': {
                    'filename': 'A/_set_1/IMG_0023.jpeg',
                    'annotations': [
                        {
                            'id': 'd8f396fa-d5c4-41e0-befc-1eff23210315',
                            'x': 0.4434,
                            'y': 0.3121,
                            'width': 0.754,
                            'height': 0.424,
                            'label': '745edd1d-80bc-4060-980a-962ab85e0268',
                            'unsure': True
                        },
                        {
                            'id': '66f3f594-01da-4a9c-820c-e69453145e71',
                            'x': 0.0134,
                            'y': 0.9943,
                            'width': 0.2343,
                            'height': 0.3,
                            'label': '5af0a711-db9d-42f4-8316-a2f729a3d8d0',
                            'unsure': False
                        },
                        # etc.
                    ]
                },

                'bc16e215-3c4e-4607-990b-b2415dd51bca': {
                    'filename': 'A/_set_2/DSC_0321.JPG',
                    'annotations': []
                },

                # etc.
            },

            'labelClasses': [
                '745edd1d-80bc-4060-980a-962ab85e0268': {
                    'name': 'Elephant',
                    'color': '#929292'
                },
                '5af0a711-db9d-42f4-8316-a2f729a3d8d0': {
                    'name': 'Giraffe',
                    'color': None
                }
            ]
        }
    ```

    Notes:
    * All images to train on are placed with their stringified UUID as a key under section 'images'.
    * Likewise, annotations belonging to an image are collated in a list under the very image's 'annotations' section.
    * Label classes are placed in a dedicated section with their identifier as key.
    * Annotations link to the label classes through the 'label' entry.
    * Any value is optional. For example, annotations may not have coordinates (if image labels), images may not have any annotations, etc. As such, you have to expect certain values to be `None`, or not be present at all.
    * For coordinates (points, bounding boxes, etc.):
        * All values are relative w.r.t. the image bounds. For example, `x = 0.5` denotes that the x coordinate of this very annotation is exactly in the middle of the image; `width = 0.23` means that the width of the annotation is 23% of the image's width, etc.
        * For bounding boxes, x and y denote the rectangle _center_.


#### Inference parameters
* **stateDict:** See above (section "Training parameters").
* **data:** Similarly to the training data, also the inference data come in Python dict format. As a matter of fact they look exactly the same as in the training case, except that all 'annotations' sections are completely missing.
* **(return value)** In the inference case, the model needs to return a Python dict containing model predictions.
These may look as follows:
```python
    return {
        '83d7b609-e3d1-45fb-8701-79f50d25087c': {
            'predictions': [
                'x': 0.3112,
                'y': 0.3322,
                'width': 0.533323454,
                'height': 0.4135,
                'label': '5af0a711-db9d-42f4-8316-a2f729a3d8d0',
                'confidence': 0.958,
                'logits': [
                    0.042,
                    0.958
                ]
            ]
        },

        'bc16e215-3c4e-4607-990b-b2415dd51bca': {
            'predictions': []
        },

        # etc.
    }
```

  Notes:
  * You need to return one entry per image, with the stringified image UUID as a dict key (you receive the image UUID and its filename through the `data` variable).
  * Every image returned must have a 'predictions' section, holding a list of predictions
  * Only the following data types are allowed for variables holding primitives: `int`, `float`, `str`. Do not provide Numpy arrays (or values), Torch tensors, etc. Always be sure to call the appropriate methods to extract raw values, such as `.toList()`, `.item()`, etc. Failing to do so will result in the _AIWorker_ trying to commit to the database, but not doing so silently (it will just put an error message to the command line console).
  * Each item in the predictions list may contain the following variables:
    * `label`: a stringified UUID of the predicted label class
    * `confidence`: a float denoting the model confidence value of the prediction
    * `logits` (optional): a list of floats of class logits, used by the `Breaking Ties` Active Learning criterion
    * Any variable required through the 'predictionType' setting in the [configuration *.ini file](configure_settings.md). For example, if 'predictionType' is set to 'boundingBoxes', you also have to provide `x`, `y`, `width` and `height` values.



## Implement a custom ranking model

Custom rankers can be used to implement more sophisticated Active Learning criteria.
All a ranker does is to accept a list of images and (fresh) predictions, to return a `priority` float value for each prediction (note: prediction, not image) that specifies how important the image with the respective prediction(s) is.

The following snippet shows a code shell for bare rankers:
```python

    class MyCustomRanker:

        def __init__(self, config, dbConnector, fileServer, options):
            """
                Ranker constructor. This is called by both the AIWorker and AIController
                modules when starting up.
                Args:
                    config: Configuration for the current AIDE project
                    dbConnector: Access to the project database
                    fileServer: Access to the instance storing the images
                    options: A custom set of options in JSON format for this ranker
            """
            self.config = config
            self.dbConnector = dbConnector
            self.fileServer = fileServer
            self.options = options


        def rank(self, data, **kwargs):
            """
                Ranking function.
                Args:
                    data: a dict object containing images and predictions made by the model
                    kwargs: optional keyword arguments provided by the AIWorker

                Returns:
                    data: the same dict object as provided as input, but with an extra
                          'priority' entry for each prediction
            """
```

Notes:
* The ranker constructor is exactly of the same format as the model constructor above.
* The 'options' argument in the constructor provides parameters as a Python dict specific to the ranker. Like for the model, the ranker parameters can be provided through a JSON file (requires setting the 'al_criterion_options_path' property of the [configuration *.ini file](configure_settings.md)).
* `data` are formatted exactly the same as provided by the model through the `inference` function above.
* All the ranker has to do in the `rank` function is to append a `float` variable 'priority' to each entry in the data's 'predictions'.
* 'priority' values must be floating points, with  higher priority being assigned to higher values. It is recommended, but not required, to limit the priority values to the `[0, 1]` range.



## General tips and tricks

### Progress and status updates

Since model training and inference are likely to be long-running tasks, you are advised to regularly post progress updates to the _AIController_ instance. This can be done by using AIDE's task queue [Celery](http://www.celeryproject.org/).

For example, the following snippet sets the job status to "in progress" and makes the AIDE interface display a message "predicting" as well as a progress bar:
```python

    from celery import current_task

    # place this anywhere in your model's train, inference, or other functions
    current_task.update_state(state='PROGRESS',
                              message='predicting',
                              meta={
                                  'done': 34,
                                  'total': 100
                              })
```
Note that `done` and `total` need to be present if the progress bar is to be filled only partially. Values need to be integers, but the maximum is not limited to 100 (e.g., you can set it to the total number of images).

`state` may take one of the following values:
* `PROGRESS`: This shows a partially filled progress bar; requires the `done` and `total` values to be set in the `meta` argument.
* `SUCCESS`, `FAILURE`: These status values indicate a completed or failed task. Avoid them and instead raise an Exception if you encounter an unsolvable problem in your AI model (the _AIWorker_ will take care of catching Exceptions and reporting completed and failed tasks).
* (anything else): Other values show an indefinite progress bar.


### Debugging your model

To debug your model in the platform itself, you can make use of Celery's built-in debugger [rdb](https://docs.celeryproject.org/en/latest/userguide/debugging.html):
``` python

    from celery.contrib import rdb

    # place this wherever you would like to set a breakpoint
    rdb.set_trace()
```

You then need to telnet into the _AIWorker_ instance to debug the task. The rdb debugger then behaves like [pdb](https://docs.python.org/3/library/pdb.html).



### Pitfalls

#### Avoiding CUDA initialization errors

These errors occur when a library using CUDA, such as PyTorch, tries to initialize the framework twice.
A very common mistake that induces this behavior is to use a CUDA-related function in the AI model constructor and then again in one of the main functions (train, inference, etc.).
**Solution:** Avoid calling the CUDA backend in any way in the constructors.
