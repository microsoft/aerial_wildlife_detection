'''
    Subclass of Detectron2's Checkpointer, able to load
    model states from an in-memory Python dict according
    to states as saved in AIDE.

    2020 Benjamin Kellenberger
'''

from detectron2.checkpoint import DetectionCheckpointer
from typing import List, Optional


class DetectionCheckpointerInMem(DetectionCheckpointer):

    def loadFromObject(self, stateDict: dict, checkpointables: Optional[List[str]] = None) -> object:
        '''
            Customized routine that loads a model state dict
            from an object, rather than a file path.
            Most of the remaining code is just copied from
            https://detectron2.readthedocs.io/_modules/fvcore/common/checkpoint.html#Checkpointer.load
        '''
        if stateDict is None or 'model' not in stateDict:
            # nothing to load; return
            return {}
        
        incompatible = self._load_model(stateDict)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in stateDict:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore

        # return any further checkpoint data
        return stateDict