/*
    Maintains the data entries currently on display.

    2019-20 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = [];
        this.numImagesPerBatch = window.numImagesPerBatch;

        this.undoStack = [];
        this.redoStack = [];

        this.skipConfirmationDialog = window.getCookie('skipAnnotationConfirmation');

        // prepare user statistics (e.g. browser)
        this._navigator = {};
        for (var i in navigator) this._navigator[i] = navigator[i];
        // this._navigator = JSON.stringify(this._navigator);

        // check if user has finished labeling
        if(window.annotationType === 'segmentationMasks') {
            // re-check if finished after every batch in this case
            this.recheckInterval = 1;
        } else {
            this.recheckInterval = Math.max(8, 64 / window.numImagesPerBatch);
        }
        this.numBatchesSeen = 0;
        this._check_user_finished();
    }

    _check_user_finished() {
        var self = this;
        $.ajax({
            url: 'getUserFinished',
            method: 'GET',
            success: function(response) {
                if(response.hasOwnProperty('finished') && response['finished']) {
                    // show message
                    $('#footer-message-panel').html('Congratulations, you have finished labeling this dataset!')
                    $('#footer-message-panel').css('color', 'green');
                    $('#footer-message-panel').show();

                    // disable querying
                    self.numBatchesSeen = -1;
                } else {
                    // reset counter for querying
                    self.numBatchesSeen = 0;
                }
            }
        })
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

                if(Object.keys(data['entries']).length === 0) {
                    // no more images to show
                    self._check_user_finished();
                    return;
                }

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

                // re-check if finished (if threshold exceeded)
                if(self.numBatchesSeen >= 0) { 
                    self.numBatchesSeen += 1;
                    if(self.numBatchesSeen >= self.recheckInterval) {
                        // re-check
                        self._check_user_finished();
                    }
                }
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self._loadNextBatch).bind(self));
                }
            }
        });
    }



    _loadReviewBatch() {
        var self = this;

        // get properties
        var minTimestamp = parseFloat($('#review-timerange').val());  ///1000;
        var skipEmptyImgs = $('#review-skip-empty').prop('checked');
        var userNames = [];
        if(window.uiControlHandler.hasOwnProperty('reviewUsersTable')) {
            // user is admin; check which names are selected
            window.uiControlHandler.reviewUsersTable.children().each(function() {
                var checked = $(this).find(':checkbox').prop('checked');
                if(checked) {
                    userNames.push($(this).find('.user-list-name').html());
                }
            })
        }

        var url = 'getImages_timestamp';
        return $.ajax({
            url: url,
            method: 'POST',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            data: JSON.stringify({
                minTimestamp: minTimestamp,
                users: userNames,
                skipEmpty: skipEmptyImgs,
                limit: this.numImagesPerBatch
            }),
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

                    // update min and max timestamp
                    var nextTimestamp = data['entries'][d]['last_checked'];
                    minTimestamp = Math.max(minTimestamp, nextTimestamp+1);
                }

                // update present classes list
                self.updatePresentClasses();

                // update slider and date text
                $('#review-timerange').val(Math.min($('#review-timerange').prop('max'), minTimestamp));
                $('#review-time-text').html(new Date(minTimestamp * 1000).toLocaleString());

                // adjust width of entries
                window.windowResized();
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // ask user to provide password again
                    window.verifyLogin((self._loadReviewBatch).bind(self));
                }
            }
        });
    }



    _entriesToJSON(minimal, onlyUserAnnotations) {
        // assemble entries
        var entries = {};
        for(var e=0; e<this.dataEntries.length; e++) {
            entries[this.dataEntries[e].entryID] = this.dataEntries[e].getProperties(minimal, onlyUserAnnotations);
        }

        // also append client statistics
        var meta = {
            browser: this._navigator,
            windowSize: [$(window).width(), $(window).height()],
            uiControls: {
                burstModeEnabled: window.uiControlHandler.burstMode
            }
        };

        return JSON.stringify({
            'entries': entries,
            'meta': meta
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

                var callback = function() {
                    if(self.redoStack.length > 0) {
                        var nb = self.redoStack.pop();
                        self._loadFixedBatch(nb.slice());
                    } else {
                        //TODO: temporary mode to ensure compatibility with running instances
                        try {
                            if($('#imorder-review').prop('checked')) {
                                self._loadReviewBatch();
                            } else {
                                self._loadNextBatch();
                            }
                        } catch {
                            self._loadNextBatch();
                        }
                    }
                };

                // check if annotation commitment is enabled
                var doSubmit = $('#imorder-auto').prop('checked') || $('#review-enable-editing').prop('checked');
                if(doSubmit) {
                    this._submitAnnotations().done(callback);
                } else {
                    // only go to next batch, don't submit annotations
                    callback();
                }
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

                // check if annotation commitment is enabled
                var doSubmit = $('#imorder-auto').prop('checked') || $('#review-enable-editing').prop('checked');
                if(doSubmit) {
                    this._submitAnnotations().done(function() {
                        self._loadFixedBatch(pb.slice());
                    });
                } else {
                    // only go to next batch, don't submit annotations
                    self._loadFixedBatch(pb.slice());
                }
                // if(dontCommit) {
                //     self._loadFixedBatch(pb.slice());

                // } else {
                //     this._submitAnnotations().done(function() {
                //         self._loadFixedBatch(pb.slice());
                //     });
                // }
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