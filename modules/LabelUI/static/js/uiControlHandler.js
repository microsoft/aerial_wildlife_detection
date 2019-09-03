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

        this.burstMode = window.getCookie('burstModeEnabled');             // if true, add and remove annotation buttons stay active upon user interaction
        if(this.burstMode === undefined) this.burstMode = false;
        else this.burstMode = window.parseBoolean(this.burstMode);

        // tools for semantic segmentation - ignored by others
        this.segmentation_properties = {
            brushType: 'rectangle',
            brushSize: 20,
            opacity: 0.75
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
            var addAnnoBtn = $('<button id="add-annotation" class="btn btn-sm btn-primary" title="Add Annotation (W)">+</button>');
            addAnnoBtn.click(addAnnoCallback);
            var removeAnnoBtn = $('<button id="remove-annotation" class="btn btn-sm btn-primary" title="Remove Annotation (R)">-</button>');
            removeAnnoBtn.click(removeAnnoCallback);
            this.staticButtons[ACTIONS.ADD_ANNOTATION] = addAnnoBtn;
            this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = removeAnnoBtn;
            dtControls.append(addAnnoBtn);
            dtControls.append(removeAnnoBtn);

            // burst mode checkbox
            var burstModeCallback = function() {
                var chkbx = $('#burst-mode-check');
                self.burstMode = !self.burstMode;
                chkbx.prop('checked', self.burstMode);
                window.setCookie('burstModeEnabled', self.burstMode);
            };
            var burstModeCheck = $('<input type="checkbox" id="burst-mode-check" class="inline-control" style="margin-right:2px" title="Enable burst mode (M)" />');
            burstModeCheck.change(burstModeCallback);
            burstModeCheck.prop('checked', this.burstMode);
            dtControls.append(burstModeCheck);
            var burstModeLabel = $('<label for="#burst-mode-check" class="inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="Enable burst mode (M)">burst mode</label>');
            burstModeLabel.click(burstModeCallback);
            dtControls.append(burstModeLabel);

            if(window.enableEmptyClass) {
                var clearAllCallback = function() {
                    self.dataHandler.clearLabelInAll();
                }
                var clearAllBtn = $('<button class="btn btn-sm btn-warning" id="clearAll-button" title="Clear all Annotations (C)">Clear All</button>');
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
            var labelAllBtn = $('<button class="btn btn-sm btn-primary" id="labelAll-button" title="Assign label to all Annotations (A)">Label All</button>');
            labelAllBtn.click(labelAllCallback);
            dtControls.append(labelAllBtn);
            var unsureBtn = $('<button class="btn btn-sm btn-warning" id="unsure-button" title="Toggle Unsure flag for Annotation (U)">Unsure</button>');
            unsureBtn.click(unsureCallback);
            dtControls.append(unsureBtn);
        }

        // semantic segmentation controls
        if(window.annotationType === 'segmentationMasks') {
            this.segmentation_controls = {
                brush_rectangle: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="static/img/controls/rectangle.svg" style="height:18px" title="Square brush" /></button>'),
                brush_circle: $('<button class="btn btn-sm btn-secondary inline-control"><img src="static/img/controls/circle.svg" style="height:18px" title="Circular brush" /></button>'),
                brush_size: $('<input class="inline-control" type="number" min="1" max="255" value="20" title="Brush size" style="width:50px" />'),
                opacity: $('<input class="inline-control" type="range" min="0" max="255" value="220" title="Segmentation opacity" style="width:100px" />')        //TODO: make available for other annotation types as well?
            };  //TODO: ranges, default

            this.segmentation_controls.brush_rectangle.click(function() {
                self.setBrushType('rectangle');
            });
            this.segmentation_controls.brush_circle.click(function() {
                self.setBrushType('circle');
            });
            this.segmentation_controls.brush_size.on({
                change: function() {
                    var val = Math.max(1, Math.min(255, this.value));
                    $(this).val(val);
                    self.setBrushSize(val);
                },
                focusin: function() {
                    window.shortcutsDisabled = true;
                },
                focusout: function() {
                    window.shortcutsDisabled = false;
                }
            });
            this.segmentation_controls.opacity.on({
                input: function() {
                    var val = Math.max(0, Math.min(255, this.value));
                    $(this).val(val);
                    self.setSegmentationOpacity(parseInt(val)/255.0);
                }
            });

            var segControls = $('<div class="inline-control"></div>');
            segControls.append(this.segmentation_controls.brush_rectangle);
            segControls.append(this.segmentation_controls.brush_circle);
            segControls.append($('<span style="margin-left:10px;margin-right:5px;color:white">Size:</span>'));
            segControls.append(this.segmentation_controls.brush_size);
            segControls.append($('<span style="margin-left:5px;color:white">px</span>'));
            segControls.append($('<span style="margin-left:10px;margin-right:5px;color:white">Opacity:</span>'));
            segControls.append(this.segmentation_controls.opacity);
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

                } else if(ccode === 'F') {
                    $('#labelclass-search-box').focus();

                } else if(ccode === 'I') {
                    self.setAction(ACTIONS.ZOOM_IN);

                } else if(ccode === 'M') {
                    if(burstModeCallback)
                        burstModeCallback();

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


        // Annotation Reviewing
        if(!window.demoMode) {
            // set up slider range
            var initSliderRange = function() {
                var slider = $('#review-timerange');
                var dateSpan = $('#review-time-text');
                return $.ajax({
                    url: 'getTimeRange',
                    method: 'POST',
                    contentType: "application/json; charset=utf-8",
                    dataType: 'json',
                    data: JSON.stringify({
                        // users: [],  //TODO: implement for admins
                        skipEmpty: $('#review-skip-empty').prop('checked')
                    }),
                    success: function(data) {
                        slider.prop('min', parseInt(data['minTimestamp'])-1);
                        slider.prop('max', parseInt(data['maxTimestamp'])+1);
                        slider.val(parseInt(data['minTimestamp'])-1);
                        dateSpan.html(new Date(data['minTimestamp']*1000 - 1).toLocaleString());
                    }
                });
            };
            var onChange = function() {
                if($('#imorder-review').prop('checked')) {
                    $('#review-controls').slideDown();
                    initSliderRange().done(function() {
                        window.dataHandler.nextBatch();
                    });
                } else {
                    $('#review-controls').slideUp();
                    window.dataHandler.nextBatch();
                }
            }
            $('#imorder-auto').change(onChange);
            $('#imorder-review').change(onChange);
            $('#review-skip-empty').change(onChange);

            $('#review-timerange').on({
                'input': function() {
                    var timestamp = parseInt($('#review-timerange').val());
                    $('#review-time-text').html(new Date(timestamp * 1000).toLocaleString());
                },
                'change': function() {
                    if($('#imorder-review').prop('checked')) {
                        window.dataHandler.nextBatch();
                    }
                }
            });

            // show user list if admin
            if(window.getCookie('isAdmin') === 'y') {
                var uTable = $('<table class="limit-users-table"><thead><tr><td><input type="checkbox" id="review-users-checkall" /></td><td>Name</td></tr></thead></table>');
                this.reviewUsersTable = $('<tbody></tbody>');
                uTable.append(this.reviewUsersTable);

                // get all the users
                $.ajax({
                    url: 'getUserNames',
                    method: 'POST',
                    contentType: "application/json; charset=utf-8",
                    dataType: 'json',
                    success: function(data) {
                        if(data.hasOwnProperty('users')) {
                            for(var u=0;u<data['users'].length; u++) {
                                var uName = data['users'][u];
                                if(uName === $('#navbar-user-dropdown').html()) {
                                    continue;
                                }
                                self.reviewUsersTable.append(
                                    $('<tr><td><input type="checkbox" /></td><td class="user-list-name">' + uName + '</td></tr>')
                                );
                            }
                            self.reviewUsersTable.find(':checkbox').change(onChange);
                            $('#review-controls').append($('<div style="margin-top:10px">View other user annotations:</div>'));
                            $('#review-controls').append(uTable);
                            $('#review-controls').append($('<div style="color:gray;font-size:9pt;font-style:italic;">If no user is selected, only your own annotations are shown.</div>'));

                            var checkAll = function() {
                                var isChecked = $('#review-users-checkall').prop('checked');
                                self.reviewUsersTable.find(':checkbox').each(function() {
                                    $(this).prop('checked', isChecked);
                                });
                                onChange();
                            }
                            $('#review-users-checkall').change(checkAll);
                        }
                    }
                });
            }

            initSliderRange();
        }
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

    setSegmentationOpacity(value) {
        value = Math.max(0, Math.min(1, value));
        this.segmentation_properties.opacity = value;
        this.renderAll();
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