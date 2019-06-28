/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = [];
        this.numImagesPerBatch = window.numImages_x * window.numImages_y;

        this.history = [];

        this._setup_controls();
    }

    _setup_controls() {

        var parentElement = $('#interface-controls');
        var self = this;

        if(window.annotationType === 'labels') {
            // assign/remove all labels buttons
            parentElement.append($('<button class="btn btn-primary" onclick="window.dataHandler.assignLabelToAll()">Label All</button>'));
            $(window).keyup(function(event) {
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
                    if(String.fromCharCode(event.which) === 'C') {
                        self.clearLabelInAll();
                    }
                });
            }

        } else {
            // add and remove buttons
            parentElement.append($('<button class="btn btn-primary" onclick="window.interfaceControls.action=window.interfaceControls.actions.ADD_ANNOTATION;">+</button>'));
            parentElement.append($('<button class="btn btn-primary" onclick="window.interfaceControls.action=window.interfaceControls.actions.REMOVE_ANNOTATIONS;">-</button>'));

            $(window).keyup(function(event) {
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
        parentElement.append($('<button class="btn btn-primary float-left" onclick="window.dataHandler.previousBatch()">Previous</button>'));
        $(window).keyup(function(event) {
            if(event.which === 37) {
                // left arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.previousBatch();
            }
        });

        parentElement.append($('<button class="btn btn-primary float-right" onclick="window.dataHandler.submitAnnotations()">Next</button>'));
        $(window).keyup(function(event) {
            if(event.which === 39) {
                // right arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.submitAnnotations();
            }
        });


        // hide predictions (annotations) if shift (ctrl) key held down
        $(window).keydown(function(event) {
            if(event.which === 16) {
                self.setPredictionsVisible(false);
            } else if(event.which === 17) {
                self.setAnnotationsVisible(false);
            }
        });
        $(window).keyup(function(event) {
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
        if(window.annotationType != 'labels') return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setLabel(window.labelClassHandler.getActiveClassID());
        }
    }

    clearLabelInAll() {
        /*
            For classification entries only: remove all assigned labels
            (if 'enableEmptyClass' is true).
        */
        if(window.annotationType != 'labels' || !window.enableEmptyClass) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setLabel(null);
        }
    }

    removeActiveAnnotations() {
        if(window.annotationType == 'labels') {
            this.clearLabelInAll();
        } else {
            for(var i=0; i<this.dataEntries.length; i++) {
                this.dataEntries[i].removeActiveAnnotations();
            }
        }
    }

    setPredictionsVisible(visible) {
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setPredictionsVisible(visible);
        }
    }

    setAnnotationsVisible(visible) {
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setAnnotationsVisible(visible);
        }
    }


    loadNextBatch() {
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
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // redirect to login page
                    window.location.href = '/';
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
        var self = this;
        var entries = this._entriesToJSON(true, false);
        $.ajax({
            url: 'submitAnnotations',
            type: 'POST',
            contentType: 'application/json; charset=utf-8',
            data: entries,
            dataType: 'json',
            success: function(response) {
                // check status
                if(response['status'] == 0) {

                    // add current image IDs to history
                    var historyEntry = [];
                    for(var i=0; i<self.dataEntries.length; i++) {
                        historyEntry.push(self.dataEntries[i]['entryID']);
                    }
                    self.history.push(historyEntry);

                    // load next batch
                    self.loadNextBatch();

                } else {
                    // error
                    //TODO: make proper messaging system
                    alert('Error: ' + response['message']);
                }
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // redirect to login page
                    window.location.href = '/';

                } else {
                    // error
                    //TODO: make proper messaging system
                    alert('Unexpected error: ' + error);
                }
            }
        });
    }


    previousBatch() {
        if(this.history.length == 0) return;
        
        var prevBatch = this.history.pop();
        
        var self = this;

        //TODO: check if changed and then submit current annotations first

        $.ajax({
            url: 'getImages',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            data: JSON.stringify({'imageIDs':prevBatch}),
            type: 'POST',
            success: function(data) {

                // clear current entries
                self.parentDiv.empty();
                self.dataEntries = [];

                for(var d in prevBatch) {
                    var entryID = prevBatch[d];
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
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // redirect to login page
                    window.location.href = '/';
                }
            }
        });
    }
}