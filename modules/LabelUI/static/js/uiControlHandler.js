/*
    Manages functionalities around the UI/viewport/canvas,
    such as control buttons (select, add, zoom, etc.).

    2019 Benjamin Kellenberger
*/


const ACTIONS = {
    DO_NOTHING: 0,
    ADD_ANNOTATION: 1,
    REMOVE_ANNOTATIONS: 2,
    ZOOM_IN: 3,
    ZOOM_OUT: 4,
    ZOOM_AREA: 5,
    PAN: 6
}

const CURSORS = [
    'pointer',
    'crosshair',
    'crosshair',
    'zoom-in',      //TODO: doesn't work with Firefox and Chrome
    'zoom-out',     //ditto
    'zoom-in',      //ditto
    'grab'
]

class UIControlHandler {

    constructor(dataHandler) {
        this.dataHandler = dataHandler;
        this.action = ACTIONS.DO_NOTHING;
        this.showLoupe = false;

        this.default_cursor = 'pointer';    // changes depending on action

        // tools for semantic segmentation - ignored by others
        this.segmentation_properties = {
            brushType: 'rectangle',
            brushSize: 20
        };

        this._setup_controls();
    }


    _setup_controls() {

        var self = this;
        this.staticButtons = {};    // buttons that can be pressed in and left that way

        /*
            viewport controls
        */
        var vpControls = $('#viewport-controls');

        // select
        var selectButton = $('<button id="select-button" class="btn btn-sm btn-secondary active" title="Select (S)"><img src="static/img/controls/select.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.DO_NOTHING] = selectButton;
        vpControls.append(selectButton);
        selectButton.click(function() {
            self.setAction(ACTIONS.DO_NOTHING);
        });

        // pan
        var panButton = $('<button id="pan-button" class="btn btn-sm btn-secondary" title="Pan (P)"><img src="static/img/controls/pan.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.PAN] = panButton;
        vpControls.append(panButton);
        panButton.click(function() {
            self.setAction(ACTIONS.PAN);
        });

        // loupe
        vpControls.append($('<button id="loupe-button" class="btn btn-sm btn-secondary" title="Toggle Loupe (B)"><img src="static/img/controls/loupe.svg" style="height:18px" /></button>'));
        $('#loupe-button').click(function(e) {
            e.preventDefault();
            self.toggleLoupe();
        });

        // zoom buttons
        vpControls.append($('<button id="zoom-in-button" class="btn btn-sm btn-secondary" title="Zoom In (I)"><img src="static/img/controls/zoom_in.svg" style="height:18px" /></button>'));
        $('#zoom-in-button').click(function() {
            self.setAction(ACTIONS.ZOOM_IN);
        });
        vpControls.append($('<button id="zoom-out-button" class="btn btn-sm btn-secondary" title="Zoom Out (O)"><img src="static/img/controls/zoom_out.svg" style="height:18px" /></button>'));
        $('#zoom-out-button').click(function() {
            self.setAction(ACTIONS.ZOOM_OUT);
        });

        var zoomAreaButton = $('<button id="zoom-area-button" class="btn btn-sm btn-secondary" title="Zoom to Area (Z)"><img src="static/img/controls/zoom_area.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.ZOOM_AREA] = zoomAreaButton;
        vpControls.append(zoomAreaButton);
        zoomAreaButton.click(function() {
            self.setAction(ACTIONS.ZOOM_AREA);
        });
        vpControls.append($('<button id="zoom-reset-button" class="btn btn-sm btn-secondary" title="Original Extent (E)"><img src="static/img/controls/zoom_extent.svg" style="height:18px" /></button>'));
        $('#zoom-reset-button').click(function() {
            self.resetZoom();
        });

        /*
            data controls
        */
        var dtControls = $('#interface-controls');

        if(!(window.annotationType === 'labels')) {
            // add and remove buttons
            var addAnnoCallback = function() {
                self.setAction(ACTIONS.ADD_ANNOTATION);
            }
            var removeAnnoCallback = function() {
                self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
            }
            var addAnnoBtn = $('<button id="add-annotation" class="btn btn-sm btn-primary">+</button>');
            addAnnoBtn.click(addAnnoCallback);
            var removeAnnoBtn = $('<button id="remove-annotation" class="btn btn-sm btn-primary">-</button>');
            removeAnnoBtn.click(removeAnnoCallback);
            this.staticButtons[ACTIONS.ADD_ANNOTATION] = addAnnoBtn;
            this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = removeAnnoBtn;
            dtControls.append(addAnnoBtn);
            dtControls.append(removeAnnoBtn);

            if(window.enableEmptyClass) {
                var clearAllCallback = function() {
                    self.dataHandler.clearLabelInAll();
                }
                var clearAllBtn = $('<button class="btn btn-sm btn-warning" id="clearAll-button">Clear All</button>');
                clearAllBtn.click(clearAllCallback);
                dtControls.append(clearAllBtn);
            }
        }

        // label all and unsure buttons
        if(window.annotationType != 'segmentationMasks') {
            var labelAllCallback = function() {
                self.dataHandler.assignLabelToAll();
            }
            var unsureCallback = function() {
                self.dataHandler.toggleActiveAnnotationsUnsure();
            }
            var labelAllBtn = $('<button class="btn btn-sm btn-primary" id="labelAll-button">Label All</button>');
            labelAllBtn.click(labelAllCallback);
            dtControls.append(labelAllBtn);
            var unsureBtn = $('<button class="btn btn-sm btn-warning" id="unsure-button">Unsure</button>');
            unsureBtn.click(unsureCallback);
            dtControls.append(unsureBtn);
        }

        // semantic segmentation controls
        if(window.annotationType === 'segmentationMasks') {
            this.segmentation_controls = {
                brush_rectangle: $('<button class="btn btn-sm btn-secondary inline-control active">Rect</button>'),
                brush_circle: $('<button class="btn btn-sm btn-secondary inline-control">Circ</button>'),
                brush_size: $('<input class="inline-control" type="range" min="1" max="255" value="20" />')
            };  //TODO: ranges, default

            this.segmentation_controls.brush_rectangle.click(function() {
                self.setBrushType('rectangle');
            });
            this.segmentation_controls.brush_circle.click(function() {
                self.setBrushType('circle');
            });
            this.segmentation_controls.brush_size.on('change', function() {
                self.setBrushSize(this.value);
            });

            var segControls = $('<div class="inline-control"></div>');
            segControls.append(this.segmentation_controls.brush_rectangle);
            segControls.append(this.segmentation_controls.brush_circle);
            segControls.append($('<span>Size:</span>'));
            segControls.append(this.segmentation_controls.brush_size);
            dtControls.append(segControls);
        }


        // next and previous batch buttons
        var nextBatchCallback = function() {
            self.dataHandler.nextBatch();
        }
        var prevBatchCallback = function() {
            self.dataHandler.previousBatch();
        }
        var prevBatchBtn = $('<button id="previous-button" class="btn btn-sm btn-primary float-left">Previous</button>');
        prevBatchBtn.click(prevBatchCallback);
        dtControls.append(prevBatchBtn);
        var nextBatchBtn = $('<button id="next-button" class="btn btn-sm btn-primary float-right">Next</button>');
        nextBatchBtn.click(nextBatchCallback);
        dtControls.append(nextBatchBtn);



        /*
            Key stroke listener
        */
        $(window).keyup(function(event) {
            if(window.uiBlocked || window.shortcutsDisabled) return;
            
            if(event.which === 16) {
                // shift key
                self.dataHandler.setPredictionsVisible(true);
                self.dataHandler.setMinimapVisible(true);

            } else if(event.which === 17) {
                // ctrl key
                self.dataHandler.setAnnotationsVisible(true);
                self.dataHandler.setMinimapVisible(true);

            } else if(event.which === 27) {
                // esc key
                self.setAction(ACTIONS.DO_NOTHING);

            } else if(event.which === 37) {
                // left arrow key
                prevBatchCallback();

            } else if(event.which === 39) {
                // right arrow key
                nextBatchCallback();

            } else if(event.which === 46 || event.which === 8) {
                // Del/backspace key; remove all active annotations
                self.dataHandler.removeActiveAnnotations();

            } else {
                // decode char keys
                var ccode = String.fromCharCode(event.which);
                if(ccode === 'A') {
                    self.dataHandler.assignLabelToAll();

                } else if(ccode === 'B') {
                    self.toggleLoupe();

                } else if(ccode === 'C') {
                    self.dataHandler.clearLabelInAll();

                } else if(ccode === 'E') {
                    self.resetZoom();

                } else if(ccode === 'I') {
                    self.setAction(ACTIONS.ZOOM_IN);

                } else if(ccode === 'O') {
                    self.setAction(ACTIONS.ZOOM_OUT);
                
                } else if(ccode === 'P') {
                    self.setAction(ACTIONS.PAN);

                } else if(ccode === 'R') {
                    self.setAction(ACTIONS.REMOVE_ANNOTATIONS);

                } else if(ccode === 'S') {
                    self.setAction(ACTIONS.DO_NOTHING);

                } else if(ccode === 'U') {
                    self.dataHandler.toggleActiveAnnotationsUnsure();

                } else if(ccode === 'W') {
                    self.setAction(ACTIONS.ADD_ANNOTATION);

                } else if(ccode === 'Z') {
                    self.setAction(ACTIONS.ZOOM_AREA);
                }
            }
        });

        $(window).keydown(function(event) {
            if(window.uiBlocked || window.shortcutsDisabled) return;
            if(event.which === 16) {
                self.dataHandler.setPredictionsVisible(false);
                self.dataHandler.setMinimapVisible(false);
            } else if(event.which === 17) {
                self.dataHandler.setAnnotationsVisible(false);
                self.dataHandler.setMinimapVisible(false);
            }
        });
    }


    getAction() {
        return this.action;
    }

    setAction(action) {
        this.action = action;

        // adjust buttons
        if(this.staticButtons.hasOwnProperty(action)) {
            for(var key in this.staticButtons) {
                this.staticButtons[key].removeClass('active');
            }
            this.staticButtons[action].addClass('active');
        }

        this.default_cursor = CURSORS[this.action];
    }

    getBrushType() {
        return this.segmentation_properties.brushType;
    }

    setBrushType(type) {
        if(type === 'rectangle') {
            this.segmentation_properties.brushType = 'rectangle';
            this.segmentation_controls.brush_rectangle.addClass('active');
            this.segmentation_controls.brush_circle.removeClass('active');
        } else if(type === 'circle') {
            this.segmentation_properties.brushType = 'circle';
            this.segmentation_controls.brush_rectangle.removeClass('active');
            this.segmentation_controls.brush_circle.addClass('active');
        } else {
            throw Error('Invalid brush type ('+type+').');
        }
    }

    getBrushSize() {
        return this.segmentation_properties.brushSize;
    }

    setBrushSize(size) {
        size = Math.min(Math.max(size, 1), 255);        //TODO: max
        this.segmentation_properties.brushSize = size;
        $(this.segmentation_controls.brush_size).attr('value', size);
    }

    getDefaultCursor() {
        return this.default_cursor;
    }

    loupeVisible() {
        return this.showLoupe;
    }

    toggleLoupe() {
        this.showLoupe = !this.showLoupe;
        if(this.showLoupe) {
            $('#loupe-button').addClass('active');
        } else {
            $('#loupe-button').removeClass('active');
        }
        this.renderAll();
    }

    resetZoom() {
        this.dataHandler.resetZoom();
    }

    renderAll() {
        this.dataHandler.renderAll();
    }
}