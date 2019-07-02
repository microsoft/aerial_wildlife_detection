/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = [];
        this.numImagesPerBatch = window.numImages_x * window.numImages_y;

        this.undoStack = [];
        this.redoStack = [];

        this._setup_controls();
    }

    _setup_controls() {

        var parentElement = $('#interface-controls');
        var self = this;

        if(window.annotationType === 'labels') {
            // assign/remove all labels buttons
            parentElement.append($('<button class="btn btn-primary" onclick="window.dataHandler.assignLabelToAll()">Label All</button>'));
            $(window).keyup(function(event) {
                if(window.uiBlocked) return;
                if(String.fromCharCode(event.which) === 'A') {
                    self.assignLabelToAll();
                } else if(event.which === 46) {
                    // Del key; remove all active annotations
                    self.removeActiveAnnotations();
                }
            });

            if(window.enableEmptyClass) {
                parentElement.append($('<button class="btn btn-primary btn-warning" onclick="window.dataHandler.clearLabelInAll()">Clear All</button>'));
                $(window).keyup(function(event) {
                    if(window.uiBlocked) return;
                    if(String.fromCharCode(event.which) === 'C') {
                        self.clearLabelInAll();
                    }
                });
            }

        } else {
            // add and remove buttons
            parentElement.append($('<button id="add-annotation" class="btn btn-primary" onclick="window.interfaceControls.action=window.interfaceControls.actions.ADD_ANNOTATION;">+</button>'));
            parentElement.append($('<button id="remove-annotation" class="btn btn-primary" onclick="window.interfaceControls.action=window.interfaceControls.actions.REMOVE_ANNOTATIONS;">-</button>'));

            $(window).keyup(function(event) {
                if(window.uiBlocked) return;
                var key = String.fromCharCode(event.which);
                if(key === '+' || key === 'W') {
                    window.interfaceControls.action = window.interfaceControls.actions.ADD_ANNOTATION;
                } else if(key === '-' || key === 'R') {
                    window.interfaceControls.action = window.interfaceControls.actions.REMOVE_ANNOTATIONS;
                } else if(event.which === 46) {
                    // Del key; remove all active annotations
                    self.removeActiveAnnotations();
                }
            });
        }
        

        // next and previous batch buttons
        parentElement.append($('<button id="previous-button" class="btn btn-primary float-left" onclick="window.dataHandler.previousBatch()">Previous</button>'));
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 37) {
                // left arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.previousBatch();
            }
        });

        parentElement.append($('<button id="next-button" class="btn btn-primary float-right" onclick="window.dataHandler.nextBatch()">Next</button>'));
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 39) {
                // right arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.nextBatch();
            }
        });

        // block previous/next buttons until everything is fully loaded
        //TODO: also do for interface controls (e.g. bbox removal)
        window.uiBlocked = false;     // false to enable loading of initial batch


        // hide predictions (annotations) if shift (ctrl) key held down
        $(window).keydown(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 16) {
                self.setPredictionsVisible(false);
            } else if(event.which === 17) {
                self.setAnnotationsVisible(false);
            }
        });
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 16) {
                self.setPredictionsVisible(true);
            } else if(event.which === 17) {
                self.setAnnotationsVisible(true);
            } else if(event.which === 27) {
                // Esc key; cancel ongoing operation
                window.interfaceControls.action = window.interfaceControls.actions.DO_NOTHING;
            }
        });
    }


    assignLabelToAll() {
        /*
            For classification entries only: assigns the selected label
            to all data entries.
        */
        if(window.uiBlocked || window.annotationType != 'labels') return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setLabel(window.labelClassHandler.getActiveClassID());
        }
    }

    clearLabelInAll() {
        /*
            For classification entries only: remove all assigned labels
            (if 'enableEmptyClass' is true).
        */
        if(window.uiBlocked || window.annotationType != 'labels' || !window.enableEmptyClass) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setLabel(null);
        }
    }

    removeActiveAnnotations() {
        if(window.uiBlocked) return;
        if(window.annotationType == 'labels') {
            this.clearLabelInAll();
        } else {
            for(var i=0; i<this.dataEntries.length; i++) {
                this.dataEntries[i].removeActiveAnnotations();
            }
        }
    }

    setPredictionsVisible(visible) {
        if(window.uiBlocked) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setPredictionsVisible(visible);
        }
    }

    setAnnotationsVisible(visible) {
        if(window.uiBlocked) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setAnnotationsVisible(visible);
        }
    }


    loadNextBatch() {
        if(window.uiBlocked) return;
        var self = this;

        //TODO: subset
        var url = 'getLatestImages?order=unlabeled&subset=default&limit=' + this.numImagesPerBatch;
        $.ajax({
            url: url,
            dataType: 'json',
            success: function(data) {
                
                // clear current entries
                self.parentDiv.empty();
                self.dataEntries = [];

                for(var d in data['entries']) {
                    // create new data entry
                    switch(String(window.annotationType)) {
                        case 'labels':
                            var entry = new ClassificationEntry(d, data['entries'][d]);
                            break;
                        case 'points':
                            var entry = new PointAnnotationEntry(d, data['entries'][d]);
                            break;
                        case 'boundingBoxes':
                            var entry = new BoundingBoxAnnotationEntry(d, data['entries'][d]);
                            break;
                        default:
                            break;
                    }

                    // append
                    self.parentDiv.append(entry.markup);
                    self.dataEntries.push(entry);
                }
                window.uiBlocked = false;
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self.loadNextBatch).bind(self));
                }
            }
        });
    }


    _entriesToJSON(minimal, onlyUserAnnotations) {
        var entries = {};
        for(var e=0; e<this.dataEntries.length; e++) {
            entries[this.dataEntries[e].entryID] = this.dataEntries[e].getProperties(minimal, onlyUserAnnotations);
        }

        return JSON.stringify({
            'entries': entries
        })
    }


    submitAnnotations() {
        if(window.uiBlocked) return;
        var self = this;
        var entries = this._entriesToJSON(true, false);
        return $.ajax({
            url: 'submitAnnotations',
            type: 'POST',
            contentType: 'application/json; charset=utf-8',
            data: entries,
            dataType: 'json',
            success: function(response) {
                // check status
                
                if(response['status'] !== 0) {
                    // error
                    //TODO: make proper messaging system
                    alert('Error: ' + response['message']);
                    return $.Deferred();
                }
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    return window.verifyLogin((self.submitAnnotations).bind(self));

                } else {
                    // error
                    //TODO: make proper messaging system
                    alert('Unexpected error: ' + error);
                    return $.Deferred();
                }
            }
        });
    }

    _loadFixedBatch(batch) {
        if(window.uiBlocked) return;
        var self = this;

        //TODO: check if changed and then submit current annotations first
        $.ajax({
            url: 'getImages',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            data: JSON.stringify({'imageIDs':batch}),
            type: 'POST',
            success: function(data) {

                // clear current entries
                self.parentDiv.empty();
                self.dataEntries = [];

                for(var d in batch) {
                    var entryID = batch[d];
                    switch(String(window.annotationType)) {
                        case 'labels':
                            var entry = new ClassificationEntry(entryID, data['entries'][entryID]);
                            break;
                        case 'points':
                            var entry = new PointAnnotationEntry(entryID, data['entries'][entryID]);
                            break;
                        case 'boundingBoxes':
                            var entry = new BoundingBoxAnnotationEntry(entryID, data['entries'][entryID]);
                            break;
                        default:
                            break;
                    }

                    // append
                    self.parentDiv.append(entry.markup);
                    self.dataEntries.push(entry);
                }

                window.uiBlocked = false;
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    var callback = function() {
                        self._loadFixedBatch(batch);
                    }
                    window.verifyLogin((callback).bind(self));
                }
            }
        });
    }


    nextBatch() {
        if(window.uiBlocked) return;
        
        var self = this;

        // add current image IDs to history
        var historyEntry = [];
        for(var i=0; i<this.dataEntries.length; i++) {
            historyEntry.push(this.dataEntries[i]['entryID']);
        }
        this.undoStack.push(historyEntry);
        
        this.submitAnnotations().done(function() {
            if(self.redoStack.length > 0) {
                var nb = self.redoStack.pop();
                self._loadFixedBatch(nb.slice());
            } else {
                self.loadNextBatch();
            }
        });
    }


    previousBatch() {
        if(window.uiBlocked) return;
        if(this.undoStack.length === 0) return;
        
        var self = this;

        // add current image IDs to history
        var historyEntry = [];
        for(var i=0; i<this.dataEntries.length; i++) {
            historyEntry.push(this.dataEntries[i]['entryID']);
        }
        this.redoStack.push(historyEntry);

        var pb = this.undoStack.pop();

        // load
        this.submitAnnotations().done(function() {
            self._loadFixedBatch(pb.slice());
        })
    }
}