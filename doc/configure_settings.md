# Configuration of the settings INI file

The settings INI file is the primary project property access point for every AIde module. It contains parameters, addresses and some passwords and must therefore never be exposed to the public!

The settings file is divided into the following categories:

[Project]

General project settings go here.

| Name |  Values | Default value | Required | Comments |
|--------------------|---------------------------|---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| projectName |  |  | YES |  Title of the project. Will be displayed on the index and interface pages. |
| projectDescription |  |  | YES | A short description of the project in a couple of sentences. |
| enableEmptyClass | 'yes', 'no' | 'yes' |  | Whether or not to allow the "no label" class. If false, every annotation or prediction must have a label class assigned. |
| annotationType | 'labels', 'boundingBoxes' |  | YES | Type of annotations to be provided by the _user_. |
| predictionType | 'labels', 'boundingBoxes' |  | YES |  Type of annotations made by the _AI model_. Note that the model must be able to handle differing kinds of labels (i.e., it must implement a way to get training supervision from the kind of labels specified under _annotationType_). |
| box_minWidth | (numeric > 0) | 20 |  | (only needed if _annotationType_ is 'boundingBoxes') Minimum width (in pixels on the screen) for user-provided bounding boxes. Should be set to a value that avoids collapsing boxes into zero-width sizes. |
| box_minHeight | (numeric > 0) | 20 |  | (only needed if _annotationType_ is 'boundingBoxes') Minimum height  (in pixels on the screen) for user-provided bounding boxes. Should be  set to a value that avoids collapsing boxes into zero-height sizes. |


[Server]

This section contains parameters for all the individual instances' addresses.

| Name | Values | Default value | Required | Comments |
|------------------|--------------------------|---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| host | (IP address or hostname) | 0.0.0.0 | YES | This is the host IP address _of the current instance_. As such, it might need to be set differently for every machine taking part in AIde. Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| port | (numeric) | 80 | YES | Network port _of the current instance_. Again, you might want to specify custom values depending on the machine here. For example, the frontend (_LabelUI_) might run on HTTP's standard port 80, but you can e.g. route the _AIWorker_ instances through different ports here.  Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| dataServer_uri | (URI) |  | YES | URI, resp. URL of the _FileServer_ instance. Note that the instance needs to be accessible to both the users accessing the _LabelUI_ webpage, as well as to any running _AIWorker_ instance.  In URL format this may include the port number **and** the _FileServer_'s "staticfiles_uri" parameter too (see below); for example: `http://fileserver.domain.com:67742/files`. |
| aiController_uri | (URI) |  |  | The same for the _AIController_ instance. This must primarily be accessible to running _AIWorker_ instances, but the value of it is also used in the frontend to determine whether AI support is enabled or not.  In URL format this may include the port number of the  _AIController_ too; for example:  `http://aicontroller.domain.com:67743`. |



[UserHandler]

| Name | Values | Default value | Required | Comments |
|----------------------|---------------|---------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time_login | (numeric > 0) | 600 | YES |  Time (in seconds) for a session to last if the user is inactive. Upon exceeding the threshold specified here, the user is either asked to re-type their password, or else redirected to the index page. |
| create_account_token | (string) |  |  | A custom string of (preferably) random characters required to be known to users who would like to create a new account on the project page. This is to make the project semi-secret. If this value is set, the webpage to create a new account can be accessed as follows: `http://<hostname>/?d=createAccount&t=<create_account_token>`, substituting the expressions in angular brackets accordingly. If left out, a new account can be created by simply visiting:  `http://<hostname>/?d=createAccount`. |


[LabelUI]



|  |  |  |  |  |
|------------------------------|--------------------------|-----------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name | Values | Default value | Required | Comments |
| numImages_x | (numeric > 0) |  | YES | Number of images in horizontal direction to show on the interface page at a time. This follows [Bootstrap](https://getbootstrap.com/)'s 12-column layout, so numbers must be divisible accordingly. You might want to increase this number for classification tasks having images with large objects, or else set to a low value (e.g. 1) for detection projects and/or small targets. |
| numImages_y | (numeric > 0) |  | YES | Number of images in vertical direction to show on the interface page at a time. You might want to increase this number for classification tasks having  images with large objects, or else set to a low value (e.g. 1) for  detection projects and/or small targets. |
| showPredictions | 'yes' | 'no' | 'yes' |  | If set to 'yes', model predictions in/of an image _might_ be shown to the user, if all further requirements match as well (see below). |
| showPredictions_minConf | (numeric) | 0.5 |  | Minimum confidence value per prediction to be shown to the user in the interface. |
| carryOverPredictions | 'yes' | 'no' | 'no' |  | If set to 'yes', predictions _might_ get "carried over", meaning that they will automatically be converted into annotations at loading time.  Note that this also works if the type of predictions and annotations (labels, bounding boxes, etc.) differ--see below. |
| carryOverPredictions_minConf | (numeric) | 0.75 |  | Minimum confidence value per prediction to be converted to an annotation. |
| carryOverRule | 'maxConfidence' | 'mode' | 'maxConfidence' |  | Required in case when the annotation and prediction types differ; in particular in a "many-to-one" mapping (e.g. predictions = bounding boxes, annotations = labels). If set to "maxConfidence", the label class of the prediction with the highest confidence value (per image) will be used. If set to "mode", the most frequently occurring label class of all the predictions in the image will be assigned as the image-wide label. Has no effect if the annotation type is != "labels". |


[AIController]



|  |  |  |  |  |
|---------------------------|-----------------|------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name | Values | Default value | Required | Comments |
| broker_URL | (URL) | amqp://localhost | YES | URL under which the message broker (RabbitMQ, Redis, etc.) can be reached. This might include an access username, password, port and trailing specifier (e.g. queue). Refer to the individual frameworks for details. |
| result_backend | (URL) | rpc:// | YES | Backend URL under which status updates and results are fetched. If the broker type is RabbitMQ, this can be left as `rpc://`, regardless of the machine the RabbitMQ broker is running on. If it is Redis, you might need to specify the server address of the machine running Redis. |
| model_lib_path | (Python import) |  | YES | Import path of the AI model class to use. Note that the class definition of the model must be accessible from the Python path. For example: if you wish to use the built-in PyTorch RetinaNet detector: `ai.models.pytorch.detection.retinanet.RetinaNet`. |
| model_options_path | (path) |  |  | File path for a JSON-encoded file listing options for the AI model class. Depending on the implementation of the AI model, this can or can not be left empty. |
| al_criterion_lib_path | (Python import) |  | YES | Import path of the Active Learning (AL) criterion to use. Note that the class definition of the AL ranker must be accessible from the Python path. For example: the built-in criteria (Breaking Ties, max confidence) can be specified as follows: ``` ai.al.builtins.breakingties.BreakingTies ai.al.builtins.maxconfidence.MaxConfidence  ``` |
| al_criterion_options_path | (path) |  |  | File path for a JSON-encoded file listing options for the AL ranker. |
| numImages_autoTrain | (numeric) | -1 |  | Number of images that need to be annotated after the last model state creation to automatically start training the model. Large numbers may result in "higher-quality" models (exposed to more training images); small numbers can produce quick model update successions. Leave out or set to -1 to disable auto-training. |

TODO: more...