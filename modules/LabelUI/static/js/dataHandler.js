/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = [];
        this.numImagesPerBatch = window.numImagesPerBatch;  //parseInt(window.numImages_x) * parseInt(window.numImages_y);

        this.undoStack = [];
        this.redoStack = [];

        this._setup_controls();
    }

    _setup_controls() {
        var parentElement = $('#interface-controls');
        var self = this;

        if(!(window.annotationType === 'labels')) {
            // add and remove buttons
            var addAnnoCallback = function() {
                if(window.uiBlocked) return;
                window.interfaceControls.action=window.interfaceControls.actions.ADD_ANNOTATION;
            }
            var removeAnnoCallback = function() {
                if(window.uiBlocked) return;
                window.interfaceControls.action=window.interfaceControls.actions.REMOVE_ANNOTATIONS;
            }
            var addAnnoBtn = $('<button id="add-annotation" class="btn btn-sm btn-primary">+</button>');
            addAnnoBtn.click(addAnnoCallback);
            var removeAnnoBtn = $('<button id="remove-annotation" class="btn btn-sm btn-primary">-</button>');
            removeAnnoBtn.click(removeAnnoCallback);
            parentElement.append(addAnnoBtn);
            parentElement.append(removeAnnoBtn);

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

        // assign/remove all labels buttons
        if(window.annotationType != 'labels' || window.enableEmptyClass) {
            parentElement.append($('<button class="btn btn-sm btn-warning" id="clearAll-button" onclick="window.dataHandler.clearLabelInAll()">Clear All</button>'));
            $(window).keyup(function(event) {
                if(window.uiBlocked) return;
                if(String.fromCharCode(event.which) === 'C') {
                    self.clearLabelInAll();
                }
            });
        }
        
        parentElement.append($('<button class="btn btn-sm btn-primary" id="labelAll-button" onclick="window.dataHandler.assignLabelToAll()">Label All</button>'));
        parentElement.append($('<button class="btn btn-sm btn-warning" id="unsure-button" onclick="window.dataHandler.toggleActiveAnnotationsUnsure()">Unsure</button>'));
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(String.fromCharCode(event.which) === 'A') {
                self.assignLabelToAll();
            } else if(event.which === 46 || event.which === 8) {
                // Del/backspace key; remove all active annotations
                self.removeActiveAnnotations();
            } else if(String.fromCharCode(event.which) === 'U') {
                self.toggleActiveAnnotationsUnsure();
            } else if(String.fromCharCode(event.which) === 'B') {
                // loupe
                self.toggleLoupe();
            } else if(String.fromCharCode(event.which) === 'R') {
                // reset zoom
                self.resetZoom();
            }
        });


        /* viewport functionalities */
        var vpControls = $('#viewport-controls');

        // loupe
        vpControls.append($('<button id="loupe-button" class="btn btn-sm btn-secondary" title="Toggle Loupe (B)"><img src="static/img/controls/loupe.svg" style="height:18px" /></button>'));
        $('#loupe-button').click(function(e) {
            e.preventDefault();
            self.toggleLoupe();
        });

        // zoom buttons
        vpControls.append($('<button id="zoom-in-button" class="btn btn-sm btn-secondary" title="Zoom In"><img src="static/img/controls/zoom_in.svg" style="height:18px" /></button>'));
        $('#zoom-in-button').click(function() {
            window.interfaceControls.action = window.interfaceControls.actions.ZOOM_IN;
        });
        vpControls.append($('<button id="zoom-out-button" class="btn btn-sm btn-secondary" title="Zoom Out"><img src="static/img/controls/zoom_out.svg" style="height:18px" /></button>'));
        $('#zoom-out-button').click(function() {
            window.interfaceControls.action = window.interfaceControls.actions.ZOOM_OUT;
        });
        vpControls.append($('<button id="zoom-area-button" class="btn btn-sm btn-secondary" title="Zoom to Area"><img src="static/img/controls/zoom_area.svg" style="height:18px" /></button>'));
        $('#zoom-area-button').click(function() {
            window.interfaceControls.action = window.interfaceControls.actions.ZOOM_AREA;
        });
        vpControls.append($('<button id="zoom-reset-button" class="btn btn-sm btn-secondary" title="Original Extent (R)"><img src="static/img/controls/zoom_extent.svg" style="height:18px" /></button>'));
        $('#zoom-reset-button').click(function() {
            self.resetZoom();
        });


        // next and previous batch buttons
        parentElement.append($('<button id="previous-button" class="btn btn-sm btn-primary float-left" onclick="window.dataHandler.previousBatch()">Previous</button>'));
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 37) {
                // left arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.previousBatch();
            }
        });

        parentElement.append($('<button id="next-button" class="btn btn-sm btn-primary float-right" onclick="window.dataHandler.nextBatch()">Next</button>'));
        $(window).keyup(function(event) {
            if(window.uiBlocked) return;
            if(event.which === 39) {
                // right arrow key
                //TODO: confirmation request if no annotations got changed by the user?
                self.nextBatch();
            }
        });

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


    renderAll() {
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].render();
        }
    }


    assignLabelToAll() {
        /*
            For classification entries only: assigns the selected label
            to all data entries.
        */
        if(window.uiBlocked) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setLabel(window.labelClassHandler.getActiveClassID());
        }
    }

    clearLabelInAll() {
        /*
            Remove all assigned labels (if 'enableEmptyClass' is true).
        */
        if(window.uiBlocked || !window.enableEmptyClass) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].removeAllAnnotations();
        }
    }

    refreshActiveAnnotations() {
        /*
            Iterates through the data entries and sets all active annotations
            inactive, unless the globally set active data entry corresponds to
            the respective data entry's entryID.
        */
        for(var i=0; i<this.dataEntries.length; i++) {
            if(this.dataEntries[i].entryID != window.activeEntryID) {
                this.dataEntries[i].setAnnotationsInactive();
            }
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

    toggleActiveAnnotationsUnsure() {
        if(window.uiBlocked) return;
        window.unsureButtonActive = true;   // for classification entries
        var annotationsActive = false;
        for(var i=0; i<this.dataEntries.length; i++) {
            var response = this.dataEntries[i].toggleActiveAnnotationsUnsure();
            if(response) annotationsActive = true;
        }

        if(annotationsActive) window.unsureButtonActive = false;
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


    toggleLoupe() {
        window.interfaceControls.showLoupe = !window.interfaceControls.showLoupe;
        if(window.interfaceControls.showLoupe) {
            $('#loupe-button').addClass('active');
        } else {
            $('#loupe-button').removeClass('active');
        }
        this.renderAll();
    }


    resetZoom() {
        for(var e=0; e<this.dataEntries.length; e++) {
            this.dataEntries[e].viewport.resetViewport();
        }
    }


    _loadNextBatch() {
        var self = this;

        var url = 'getLatestImages?order=unlabeled&subset=default&limit=' + this.numImagesPerBatch;
        return $.ajax({
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

                //TODO: needs to be put here instead of init script
                // adjust width of entries
                window.windowResized();
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self._loadNextBatch).bind(self));
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


    submitAnnotations(silent) {
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
                    if(!silent) {
                        //TODO: make proper messaging system
                        alert('Error: ' + response['message']);
                        return $.Deferred();
                    }
                }
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    return window.verifyLogin((self.submitAnnotations).bind(self));

                } else {
                    // error
                    if(!silent) {
                        //TODO: make proper messaging system
                        alert('Unexpected error: ' + error);
                        return $.Deferred();
                    }
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

                window.setUIblocked(false);
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
                self._loadNextBatch();
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