# Configuration of the settings INI file

The settings INI file is the primary project property access point for every AIde module. It contains parameters, addresses and some passwords and must therefore never be exposed to the public!

The settings file is divided into the following categories:

## [Project]

General project settings go here.

| Name | Values | Default value | Required | Comments |
|----------------------|---------------------------|-------------------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| projectName |  |  | YES | Title of the project. Will be displayed on the index and interface pages. |
| projectDescription |  |  | YES | A short description of the project in a couple of sentences. |
| enableEmptyClass | 'yes', 'no' | 'yes' |  | Whether or not to allow the "no label" class. If false, every annotation or prediction must have a label class assigned. |
| annotationType | 'labels', 'boundingBoxes' |  | YES | Type of annotations to be provided by the _user_. |
| predictionType | 'labels', 'boundingBoxes' |  | YES | Type of annotations made by the _AI model_. Note that the model must be able to handle differing kinds of labels (i.e., it must implement a way to get training supervision from the kind of labels specified under _annotationType_). |
| box_minWidth | (numeric > 0) | 20 |  | (only needed if _annotationType_ is 'boundingBoxes') Minimum width (in pixels on the screen) for user-provided bounding boxes. Should be set to a value that avoids collapsing boxes into zero-width sizes. |
| box_minHeight | (numeric > 0) | 20 |  | (only needed if _annotationType_ is 'boundingBoxes') Minimum height  (in pixels on the screen) for user-provided bounding boxes. Should be  set to a value that avoids collapsing boxes into zero-height sizes. |
| adminName | (string) |  | YES | Username of the AIde account with administrator rights added by default to the database. |
| adminEmail | (e-mail string) |  | YES | E-mail address of the AIde administrator account. |
| adminPassword | (string) |  | YES | Plain text password of the AIde administrator account. |
| welcome_message_file | (file path) | modules/LabelUI/static/templates/welcome_message.html |  | File path to a text file containing a message to be shown in the tutorial. The message may be formatted with HTML tags and is embedded into the tutorial page, visible when the user first logs in, or else clicks the "Help" button in the top right corner. See file "config/welcome_message.html" for an example. |
| backdrops_file | (file path) | modules/StaticFiles/static/json/backdrops.json |  | File path to a JSON-formatted file containing information on the images to be shown in the background of non-interface pages (index, about, etc.). File must contain a root directory in the file system of the _LabelUI_ instance where the images are to be found, as well as an array of image names and, optionally, a string defining copyright information. See file "config/backdrops.json" for an example. |
| demoMode | 'yes', 'no' | 'no' |  | Whether to run AIde in demo mode or not. In demo mode, the platform behaves as follows: - User accounts are disabled; visitors automatically redirected to the interface page - No annotations are being stored on the server. However, annotations made by users are cached in the browser until the page is reloaded. - Upon clicking "Next", images are shown in random order - The "Log out" button is disabled and hidden - The _AIController_ module is disabled |


## [Server]

This section contains parameters for all the individual instances' addresses.

| Name | Values | Default value | Required | Comments |
|------------------|--------------------------|---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| host | (IP address or hostname) | 0.0.0.0 | YES | This is the host IP address _of the current instance_. As such, it might need to be set differently for every machine taking part in AIde. Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| port | (numeric) | 80 | YES | Network port _of the current instance_. Again, you might want to specify custom values depending on the machine here. For example, the frontend (_LabelUI_) might run on HTTP's standard port 80, but you can e.g. route the _AIWorker_ instances through different ports here.  Be sure to change the individual addresses below to make the machines reachable to each other, whenever necessary. |
| index_uri | (URI) | / |  | URL snippet under which the index page can be found. By default this can be left as "/", but may be changed if AIde is e.g. deployed under a sub-URL, such as "http://www.mydomain.com/aide", in which case it would have to be changed to "/aide". |
| dataServer_uri | (URI) |  | YES | URI, resp. URL of the _FileServer_ instance. Note that the instance needs to be accessible to both the users accessing the _LabelUI_ webpage, as well as to any running _AIWorker_ instance.  In URL format this may include the port number **and** the _FileServer_'s "staticfiles_uri" parameter too (see below); for example: `http://fileserver.domain.com:67742/files`. |
| aiController_uri | (URI) |  |  | The same for the _AIController_ instance. This must primarily be accessible to running _AIWorker_ instances, but the value of it is also used in the frontend to determine whether AI support is enabled or not.  In URL format this may include the port number of the  _AIController_ too; for example:  `http://aicontroller.domain.com:67743`. |



## [UserHandler]

| Name | Values | Default value | Required | Comments |
|----------------------|---------------|---------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| time_login | (numeric > 0) | 600 | YES |  Time (in seconds) for a session to last if the user is inactive. Upon exceeding the threshold specified here, the user is either asked to re-type their password, or else redirected to the index page. |
| create_account_token | (string) |  |  | A custom string of (preferably) random characters required to be known to users who would like to create a new account on the project page. This is to make the project semi-secret. If this value is set, the webpage to create a new account can be accessed as follows: `http://<hostname>/?d=createAccount&t=<create_account_token>`, substituting the expressions in angular brackets accordingly. If left out, a new account can be created by simply visiting:  `http://<hostname>/?d=createAccount`. |


## [AIController]

| Name | Values | Default value | Required | Comments |
|-------------------------|-----------|--------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| broker_URL | (URL) | amqp://localhost | YES | URL under which the message broker (RabbitMQ, Redis, etc.) can be reached. This might include an access username, password, port and trailing specifier (e.g. queue). Refer to the individual frameworks for details. |
| result_backend | (URL) | redis://localhost:6379/0 | YES | Backend URL under which status updates and results are fetched. **Important:** in general, and especially if AIde is to be [deployed](deployment.md), the _AIController_ instance is restarted or wrapped in a multi-threaded server, it is required to use a persistent backend for the message store. Do not use `rpc` in this case. The recommended backend is [Redis](http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html). See details [here](#set-up-the-message-broker). |
| maxNumWorkers_train | (numeric) | -1 |  | Maximum number of AIWorker instances to consider when training. -1 means that all available AIWorkers will be involved in training, and that the images will be distributed evenly across them. If > 1 or = -1, the training images will be distributed evenly over the number of AIWorkers specified, and the model's 'average_model_states' function will be called once all workers have finished training to generate a new, holistic model state. Note that this might not always be preferred (some models might not allow to be averaged). In this case, set this number to 1 to limit training (on all training images) to just one AIWorker. |
| maxNumWorkers_inference | (numeric) | -1 |  | Maximum number of AIWorker instances to involve when doing inference on images. -1 means that all available AIWorkers will be involved, and that the images will be distributed evenly across them. |
| maxNumImages_train | (numeric) |  | YES |  Maximum number of images to train on at a time. This value may be overridden by the number specified by an administrator in the UI while setting up a manual training process. |
| maxNumImages_inference | (numeric) |  | YES | Maximum number of images to do inference on at a time. This value may be overridden by the number specified by an administrator in the UI while setting up a manual training process. |


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