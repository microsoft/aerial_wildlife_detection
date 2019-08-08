# Configuration of the settings INI file

The settings INI file is the primary project property access point for every AIde module. It contains parameters, addresses and some passwords and must therefore never be exposed to the public!

The settings file is divided into the following categories:

## [Project]

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
| adminName | (string) |  | YES | Username of the AIde account with administrator rights added by default to the database. |
| adminEmail | (e-mail string) |  | YES | E-mail address of the AIde administrator account. |
| adminPassword | (string) |  | YES | Plain text password of the AIde administrator account. |
| welcome_message_file | (file path) | modules/LabelUI/static/templates/welcome_message.html |  | File path to a text file containing a message to be shown in the tutorial. The message may be formatted with HTML tags and is embedded into the tutorial page, visible when the user first logs in, or else clicks the "Help" button in the top right corner. See file "config/welcome_message.html" for an example. |
| backdrops_file | (file path) | modules/LabelUI/static/json/backdrops.json |  | File path to a JSON-formatted file containing information on the images to be shown in the background of non-interface pages (index, about, etc.). File must contain a root directory in the file system of the _LabelUI_ instance where the images are to be found, as well as an array of image names and, optionally, a string defining copyright information. See file "config/backdrops.json" for an example. |


## [Server]

This section contains parameters for all the individual instances' addresses.

| Name | Values | Default value | Required | Comments |
|------------------|--------------------------|---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| host | (IP address or hostname) | 0.0.0.0 | YES | This is the host IP address _of the current instance_. As such, it might need to be set differently for every machine taking part in AIde. Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| port | (numeric) | 80 | YES | Network port _of the current instance_. Again, you might want to specify custom values depending on the machine here. For example, the frontend (_LabelUI_) might run on HTTP's standard port 80, but you can e.g. route the _AIWorker_ instances through different ports here.  Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| dataServer_uri | (URI) |  | YES | URI, resp. URL of the _FileServer_ instance. Note that the instance needs to be accessible to both the users accessing the _LabelUI_ webpage, as well as to any running _AIWorker_ instance.  In URL format this may include the port number **and** the _FileServer_'s "staticfiles_uri" parameter too (see below); for example: `http://fileserver.domain.com:67742/files`. |
| aiController_uri | (URI) |  |  | The same for the _AIController_ instance. This must primarily be accessible to running _AIWorker_ instances, but the value of it is also used in the frontend to determine whether AI support is enabled or not.  In URL format this may include the port number of the  _AIController_ too; for example:  `http://aicontroller.domain.com:67743`. |



## [UserHandler]

| Name | Values | Default value | Required | Comments |
|----------------------|---------------|---------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time_login | (numeric > 0) | 600 | YES |  Time (in seconds) for a session to last if the user is inactive. Upon exceeding the threshold specified here, the user is either asked to re-type their password, or else redirected to the index page. |
| create_account_token | (string) |  |  | A custom string of (preferably) random characters required to be known to users who would like to create a new account on the project page. This is to make the project semi-secret. If this value is set, the webpage to create a new account can be accessed as follows: `http://<hostname>/?d=createAccount&t=<create_account_token>`, substituting the expressions in angular brackets accordingly. If left out, a new account can be created by simply visiting:  `http://<hostname>/?d=createAccount`. |


## [LabelUI]

| Name | Values | Default value | Required | Comments |
|------------------------------|-------------------------|--------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| numImagesPerBatch | (numeric > 0) | 1 |  | Number of images to load and display at a time in the user interface. |
| minImageWidth | (numeric > 0) | 300 |  | Minimum image width in pixels displayed in the user interface. The actual width of the image(s) displayed might be larger, depending on the browser window size and the number of images shown, but never smaller than the value provided here. If the browser window gets too small to display even just one image completely, scrollbars appear that require the user to pan around in the portion of the image displayed. |
| numImageColumns_max | (numeric > 0) | 1 |  | Number of columns to show at maximum, if more than one image is displayed in the user interface at a time. This provides an additional visualization constraint to control the number of visuals to be shown in the interface at a time. |
| showPredictions | 'yes' | 'no' |  | Whether to display model predictions in the user interface or not. Note that predictions might still be "carried over" (i.e., converted to annotations) if they meet the criterion defined. See next fields for details. |
| showPredictions_minConf | (numeric) | 0.5 |  | Minimum confidence value per prediction to be shown to the user in the interface. |
| carryOverPredictions | 'yes' | 'no' |  | Whether to "carry over" predictions (i.e., auto-generate annotations from them). If 'yes', predictions with a confidence of a threshold set (next field) or higher will be auto-converted into annotations. |
| carryOverPredictions_minConf | (numeric) | 0.75 |  | Minimum confidence value per prediction to be converted to an annotation. |
| carryOverRule | 'maxConfidence', 'mode' | 'mode' |  | Prediction carry-over rule for conversions across different annotation/prediction types. This applies if e.g. multiple points are to be converted to a single, image-wide label.  If set to 'maxConfidence', the label class of the prediction with the highest confidence in the image will be used. Option 'mode' creates an image label based on the most frequently occurring label class in the image. |
| styles_file | (file path) | modules/LabelUI/static/json/styles.json |  |  File path to a JSON-formatted file defining user interface styles. See file "config/styles.json" for an example. |


## [AIController]

| Name | Values | Default value | Required | Comments |
|---------------------------|-----------------|------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| broker_URL | (URL) | amqp://localhost | YES | URL under which the message broker (RabbitMQ, Redis, etc.) can be reached. This might include an access username, password, port and trailing specifier (e.g. queue). Refer to the individual frameworks for details. |
| result_backend | (URL) | redis://localhost:6379/0 | YES | Backend URL under which status updates and results are fetched. **Important:** in general, and especially if AIde is to be [deployed](deployment.md), the _AIController_ instance is restarted or wrapped in a multi-threaded server, it is required to use a persistent backend for the message store. Do not use `rpc` in this case. The recommended backend is [Redis](http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html). See details [here](#set-up-the-message-broker). |
| model_lib_path | (Python import) |  | YES | Import path of the AI model class to use. Note that the class definition of the model must be accessible from the Python path. For example: if you wish to use the built-in PyTorch RetinaNet detector: `ai.models.pytorch.boundingBoxes.retinanet.RetinaNet`. See [here](builtin_models.md) for models built-in to AIde, and [here](custom_model.md) if you wish to create and plug in your own AI model. |
| model_options_path | (path) |  |  | File path for a JSON-encoded file listing options for the AI model class. Depending on the implementation of the AI model, this can or can not be left empty. |
| al_criterion_lib_path | (Python import) |  | YES | Import path of the Active Learning (AL) criterion to use. Note that the class definition of the AL ranker must be accessible from the Python path. For example: the built-in criteria (Breaking Ties, max confidence) can be specified as follows: ``` ai.al.builtins.breakingties.BreakingTies ai.al.builtins.maxconfidence.MaxConfidence  ```. See [here](builtin_models.md) for models built-in to AIde, and [here](custom_model.md) if you wish to create and plug in your own AL criterion. |
| al_criterion_options_path | (path) |  |  | File path for a JSON-encoded file listing options for the AL ranker. |
| numImages_autoTrain | (numeric) | -1 |  | Number of images that need to be annotated after the last model state creation to automatically start training the model. Large numbers may result in "higher-quality" models (exposed to more training images); small numbers can produce quick model update successions. Leave out or set to -1 to disable auto-training. |
| minNumAnnoPerImage | (numeric) | 0 |  | Number of annotations for an image to have to be considered for model training. This is particularly useful for detection tasks with a low number of true objects to avoid model convergence to a state where it is swamped by negatives. Leave out or set to 0 to include all images, even with zero annotations. |
| maxNumImages_train | (numeric) |  | YES |  Maximum number of images to train on at a time. This value may be overridden by the number specified by an administrator in UI while setting up a manual training process. |
| maxNumImages_inference | (numeric) |  | YES | Maximum number of images to do inference on at a time. This value may be overridden by the number specified by an administrator in UI while setting up a manual training process. |
| maxNumWorkers_train | (numeric) | -1 |  | Maximum number of AIWorker instances to consider when training. -1 means that all available AIWorkers will be involved in training, and that the images will be distributed evenly across them. If > 1 or = -1, the training images will be distributed evenly over the number of AIWorkers specified, and the model's 'average_model_states' function will be called once all workers have finished training to generate a new, holistic model state. Note that this might not always be preferred (some models might not allow to be averaged). In this case, set this number to 1 to limit training (on all training images) to just one AIWorker. |
| maxNumWorkers_inference | (numeric) | -1 |  | Maximum number of AIWorker instances to involve when doing inference on images. -1 means that all available AIWorkers will be involved, and that the images will be distributed evenly across them. |


## [FileServer]

| Name | Values | Default value | Required | Comments |
|-----------------|--------------|---------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| staticfiles_dir | (path) |  | YES | Root directory on the local disk of the file server to serve files from. |
| staticfiles_uri | (URI string) |  | YES | URI snippet to append after the file server's host name. For example, if set to `/files`, the file server provides files through `http(s)://<host>:<port>/files`. |



## [Database]

| Name | Values | Default value | Required | Comments |
|---------------------|-----------|---------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name | (string) |  | YES | Name of the Postgres database on the server. |
| schema | (string) |  | YES | Schema within the database to store the project data in. |
| host | (URL) |  | YES | URL under which the database can be accessed (without the port). Can be set to `localhost` if and only if all AIde modules are to be launched on the same server the database is hosted on. |
| port | (numeric) |  | YES | Port the database listens to. Note: Postgres' default port is 5432; unless the database instance is solely connected to LAN (and not WAN), it is advised to change the Postgres port to another, free value. The [database installation instructions](setup_db.md) will automatically consider the custom port. |
| user | (string) |  | YES | Name of the user that is given access to the database. |
| password | (string) |  | YES | Password (in clear text) for the Postgres user. **NOTE:** unlike all other database fields, the password is case-sensitive. |
| max_num_connections | (numeric) | 16 |  | Maximum number of connections to the database per server running an AIde module. This number, multiplied by the number of server instances running AIde, must not exceed the maximum number of connections defined in Postgres' configuration file. |