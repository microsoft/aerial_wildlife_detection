/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = [];
        this.numImagesPerBatch = window.numImagesPerBatch;

        this.undoStack = [];
        this.redoStack = [];

        this.skipConfirmationDialog = window.getCookie('skipAnnotationConfirmation');
    }


    renderAll() {
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].render();
        }
    }


    resetZoom() {
        for(var e=0; e<this.dataEntries.length; e++) {
            this.dataEntries[e].viewport.resetViewport();
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

    setMinimapVisible(visible) {
        if(window.uiBlocked) return;
        for(var i=0; i<this.dataEntries.length; i++) {
            this.dataEntries[i].setMinimapVisible(visible);
        }
    }


    getAllPresentClassIDs() {
        /*
            Returns a dict of all label class IDs that are present
            in the image entry/entries.
        */
        var presentClassIDs = {};
        for(var key in this.dataEntries) {
            presentClassIDs = {...presentClassIDs, ...this.dataEntries[key].getActiveClassIDs()};
        }
        return presentClassIDs;
    }


    updatePresentClasses() {
        //TODO: too much of a mess right now
        // /*
        //     Examines the label classes present in the images and
        //     puts their markup to the top of the class entries.
        // */
        // var presentClassIDs = this.getAllPresentClassIDs();

        // //TODO: this is very expensive; replace with LUT on the long term...
        // var container = $('#legend-entries-active-container');
        // container.empty();
        // for(var key in presentClassIDs) {
        //     container.append(window.labelClassHandler.getClass(key).getMarkup(true));
        // }
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
                        case 'segmentationMasks':
                            var entry = new SemanticSegmentationEntry(d, data['entries'][d]);
                            break;
                        default:
                            break;
                    }

                    // append
                    self.parentDiv.append(entry.markup);
                    self.dataEntries.push(entry);
                }

                // update present classes list
                self.updatePresentClasses();

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


    _submitAnnotations(silent) {
        if(window.demoMode) {
            return $.Deferred().promise();
        }

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
                    return window.verifyLogin((self._submitAnnotations).bind(self));

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
        var self = this;

        // check if changed and then submit current annotations first
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
                        case 'segmentationMasks':
                            var entry = new SemanticSegmentationEntry(entryID, data['entries'][entryID]);
                            break;
                        default:
                            break;
                    }

                    // append
                    self.parentDiv.append(entry.markup);
                    self.dataEntries.push(entry);
                }

                // update present classes list
                self.updatePresentClasses();

                // adjust width of entries
                window.windowResized();

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

        if(window.demoMode) {
            var _next_batch = function() {
                // in demo mode we add the entire objects to the history
                for(var e=0; e<self.dataEntries.length; e++) {
                    self.dataEntries[e].markup.detach();
                }
                self.undoStack.push(self.dataEntries.slice());
                if(self.redoStack.length > 0) {
                    // re-initialize stored data entries
                    var entries = self.redoStack.pop();
                    self.dataEntries = entries;
                    for(var e=0; e<self.dataEntries.length; e++) {
                        self.parentDiv.append(self.dataEntries[e].markup);
                    }
                } else {
                    self._loadNextBatch();
                }
            }

        } else {
            var _next_batch = function() {
                // add current image IDs to history
                var historyEntry = [];
                for(var i=0; i<this.dataEntries.length; i++) {
                    historyEntry.push(this.dataEntries[i]['entryID']);
                }
                this.undoStack.push(historyEntry);
                
                this._submitAnnotations().done(function() {
                    if(self.redoStack.length > 0) {
                        var nb = self.redoStack.pop();
                        self._loadFixedBatch(nb.slice());
                    } else {
                        self._loadNextBatch();
                    }
                });
            }
        }
        this._showConfirmationDialog((_next_batch).bind(this));
    }


    previousBatch() {
        if(window.uiBlocked || this.undoStack.length === 0) return;
        
        var self = this;

        if(window.demoMode) {
            var _previous_batch = function() {
                for(var e=0; e<self.dataEntries.length; e++) {
                    self.dataEntries[e].markup.detach();
                }
                self.redoStack.push(self.dataEntries.slice());

                // re-initialize stored data entries
                var entries = self.undoStack.pop();
                self.dataEntries = entries;
                for(var e=0; e<self.dataEntries.length; e++) {
                    self.parentDiv.append(self.dataEntries[e].markup);
                }
            }
        } else {
            var _previous_batch = function() {
                // add current image IDs to history
                var historyEntry = [];
                for(var i=0; i<this.dataEntries.length; i++) {
                    historyEntry.push(this.dataEntries[i]['entryID']);
                }
                this.redoStack.push(historyEntry);
    
                var pb = this.undoStack.pop();
    
                // load
                this._submitAnnotations().done(function() {
                    self._loadFixedBatch(pb.slice());
                })
            };
        }
        this._showConfirmationDialog((_previous_batch).bind(this));
    }


    _showConfirmationDialog(callback_yes) {
        
        // go to callback directly if user requested not to show message anymore
        if(this.skipConfirmationDialog) {
            callback_yes();
            return;
        }

        // create markup
        if(this.confirmationDialog_markup === undefined) {
            this.confirmationDialog_markup = {};
            this.confirmationDialog_markup.cookieCheckbox = $('<input type="checkbox" style="margin-right:10px" />');
            var cookieLabel = $('<label style="margin-top:20px">Do not show this message again.</label>').prepend(this.confirmationDialog_markup.cookieCheckbox);
            this.confirmationDialog_markup.button_yes = $('<button class="btn btn-primary btn-sm" style="display:inline;float:right">Yes</button>');
            var button_no = $('<button class="btn btn-secondary btn-sm" style="display:inline">No</button>');
            button_no.click(function() {
                window.showOverlay(null);
            });
            var buttons = $('<div></div>').append(button_no).append(this.confirmationDialog_markup.button_yes);
            this.confirmationDialog_markup.markup = $('<div><h2>Confirm annotations</h2><div>Are you satisfied with your annotations?</div></div>').append(cookieLabel).append(buttons);
        }

        // wrap callbacks
        var self = this;
        var dispose = function() {
            // check if cookie is to be set
            var skipMsg = self.confirmationDialog_markup.cookieCheckbox.is(':checked');
            if(skipMsg) {
                window.setCookie('skipAnnotationConfirmation', true, 365);
                self.skipConfirmationDialog = true;
            }
            window.showOverlay(null);
        }
        var action_yes = function() {
            dispose();
            callback_yes();
        }
        this.confirmationDialog_markup.button_yes.click(action_yes);

        window.showOverlay(this.confirmationDialog_markup.markup, false, false);
    }
}