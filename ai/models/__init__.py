'''
    Abstract model class, providing code shells for the AIWorker.

    2019 Benjamin Kellenberger
'''

class AIModel:
    def __init__(self, project, config, dbConnector, fileServer, options):
        """
            Model constructor. This is called by both the AIWorker and AIController
            modules when starting up.
            Args:
                project: str, name of the current AIde project
                config: Configuration for the platform
                dbConnector: Access to the project database
                fileServer: Access to the instance storing the images
                options: A custom set of options in JSON format for this model
        """
        self.project = project
        self.config = config
        self.dbConnector = dbConnector
        self.fileServer = fileServer
        self.options = options


    def train(self, stateDict, data, updateStateFun):
        """
            Training function. This function gets called by each individual AIWorkers
            when the model is supposed to be trained for another round.
            Args:
                stateDict: a bytes object containing the model's current state
                data: a dict object containing the image metadata to be trained on
                updateStateFun: function handle for updating the progress to the
                                AIController
            
            Returns:
                stateDict: a bytes object containing the model's state after training
        """
        raise NotImplementedError('not implemented for base class.')
    

    def average_model_states(self, stateDicts, updateStateFun):
        """
            Averaging function. If AIde is configured to distribute training to multiple
            AIWorkers, and if multiple AIWorkers are attached, this function will be called
            by exactly one AIWorker after the "train" function has finished.
            Args:
                stateDicts: a list of N bytes objects containing model states as trained by
                            the N AIWorkers attached
                updateStateFun: function handle for updating the progress to the
                                AIController

            Returns:
                stateDict: a bytes object containing the combined model states
        """
        raise NotImplementedError('not implemented for base class.')


    def inference(self, stateDict, data, updateStateFun):
        """
            Inference function. This gets called everytime the model is supposed to be run on
            a set of images. The inference job might be distributed to multiple AIWorkers, but
            there is no need to worry about concurrency or race conditions, as each inference
            job is handled separately.
            Args:
                stateDict: a bytes object containing the latest model state
                data: a dict object containing the metadata of the images the model needs to
                        predict
                updateStateFun: function handle for updating the progress to the
                                AIController
        """
        raise NotImplementedError('not implemented for base class.')