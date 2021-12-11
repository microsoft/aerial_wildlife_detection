/*
    Manages functionalities around the UI/viewport/canvas,
    such as control buttons (select, add, zoom, etc.).

    2019-21 Benjamin Kellenberger
*/


const ACTIONS = {
    DO_NOTHING: 0,
    ADD_ANNOTATION: 1,
    REMOVE_ANNOTATIONS: 2,
    ZOOM_IN: 3,
    ZOOM_OUT: 4,
    ZOOM_AREA: 5,
    PAN: 6,

    // for segmentation and area selection
    PAINT_BUCKET: 7,
    ERASE_SELECTION: 8,
    ADD_SELECT_POLYGON: 9,
    ADD_SELECT_POLYGON_MAGNETIC: 10,
    GRAB_CUT: 11
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
        if(this.burstMode === undefined || this.burstMode === null) this.burstMode = false;
        else this.burstMode = window.parseBoolean(this.burstMode);

        this.magneticPolygon = false;       // for "polygons" annotation type, but not for area selection controls

        window.loupeMagnification = 0.11;

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
        var selectButton = $('<button id="select-button" class="btn btn-sm btn-secondary active" title="Select (S)"><img src="/static/interface/img/controls/select.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.DO_NOTHING] = selectButton;
        vpControls.append(selectButton);
        selectButton.click(function() {
            self.setAction(ACTIONS.DO_NOTHING);
        });

        // pan
        var panButton = $('<button id="pan-button" class="btn btn-sm btn-secondary" title="Pan (P)"><img src="/static/interface/img/controls/pan.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.PAN] = panButton;
        vpControls.append(panButton);
        panButton.click(function() {
            self.setAction(ACTIONS.PAN);
        });

        // zoom buttons
        vpControls.append($('<button id="zoom-in-button" class="btn btn-sm btn-secondary" title="Zoom In (I)"><img src="/static/interface/img/controls/zoom_in.svg" style="height:18px" /></button>'));
        $('#zoom-in-button').click(function() {
            self.setAction(ACTIONS.ZOOM_IN);
        });
        vpControls.append($('<button id="zoom-out-button" class="btn btn-sm btn-secondary" title="Zoom Out (O)"><img src="/static/interface/img/controls/zoom_out.svg" style="height:18px" /></button>'));
        $('#zoom-out-button').click(function() {
            self.setAction(ACTIONS.ZOOM_OUT);
        });

        var zoomAreaButton = $('<button id="zoom-area-button" class="btn btn-sm btn-secondary" title="Zoom to Area (Z)"><img src="/static/interface/img/controls/zoom_area.svg" style="height:18px" /></button>');
        this.staticButtons[ACTIONS.ZOOM_AREA] = zoomAreaButton;
        vpControls.append(zoomAreaButton);
        zoomAreaButton.click(function() {
            self.setAction(ACTIONS.ZOOM_AREA);
        });
        vpControls.append($('<button id="zoom-reset-button" class="btn btn-sm btn-secondary" title="Original Extent (E)"><img src="/static/interface/img/controls/zoom_extent.svg" style="height:18px" /></button>'));
        $('#zoom-reset-button').click(function() {
            self.resetZoom();
        });

        // loupe
        var loupeMarkup = $('<div style="background:#545b61;border-radius:5px;"></div>');
        vpControls.append(loupeMarkup);
        loupeMarkup.append($('<button id="loupe-button" class="btn btn-sm btn-secondary" title="Toggle Loupe (B)"><img src="/static/interface/img/controls/loupe.svg" style="height:18px" /></button>'));
        $('#loupe-button').click(function(e) {
            e.preventDefault();
            self.toggleLoupe();
        });
        loupeMarkup.append($('<div style="width:24px;height:60px;"><input type="range" id="loupe-zoom-range" title="Loupe magnification factor" min="1" max="15" step="any" value="11" style="width:40px;transform-origin:30px 20px;transform:rotate(-90deg)" /></div>'));
        $('#loupe-zoom-range').on('input', function(e) {
            let factor = (15 - $(this).val()) / 100;
            self.setLoupeMagnification(factor);
        });

        /*
            data controls
        */
        var dtControls = $('#interface-controls');

        if(!(['labels', 'segmentationMasks'].includes(window.annotationType))) {
            // add button
            var addAnnoCallback = function() {
                self.setAction(ACTIONS.ADD_ANNOTATION);
            }
            var addAnnoBtn = $('<button id="add-annotation" class="btn btn-sm btn-primary" title="Add Annotation (W)">+</button>');
            addAnnoBtn.click(addAnnoCallback);
            
            this.staticButtons[ACTIONS.ADD_ANNOTATION] = addAnnoBtn;
            
            dtControls.append(addAnnoBtn);

            if(window.annotationType === 'polygons') {
                // show toggle for magnetic polygon
                let magneticPolyCallback = function() {
                    var chkbx = $('#magnetic-polygon-check');
                    self.magneticPolygon = !self.magneticPolygon;
                    chkbx.prop('checked', self.magneticPolygon);
                    // window.setCookie('magneticPolygonEnabled', self.magneticPolygon);
                };
                let magneticPolygonCheck = $('<input type="checkbox" id="magnetic-polygon-check" class="custom-control-input inline-control" style="margin-right:2px" title="Magnetic Polygon" />');
                magneticPolygonCheck.change(magneticPolyCallback);
                magneticPolygonCheck.prop('checked', this.magneticPolygon);
                let magneticPolygonCheckContainer = $('<div id="magnetic-lasso-control" class="custom-control custom-switch inline-control"></div>');
                magneticPolygonCheckContainer.append(magneticPolygonCheck);
                magneticPolygonCheckContainer.append($('<label for="magnetic-polygon-check" class="custom-control-label inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="Magnetic Polygon"><img src="/static/interface/img/controls/lasso_magnetic.svg" style="height:18px" title="Magnetic Polygon" /></label>'));
                dtControls.append(magneticPolygonCheckContainer);

                // GrabCut
                let grabCutBtn = $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/grabcut.svg" style="height:18px" title="Grab Cut" /></button>');
                grabCutBtn.on('click', function() {
                    // special routine for Polygons: use GrabCut for active polygons, or else set action
                    if(self.dataHandler.getNumActiveAnnotations()) {
                        // at least one element is active; perform GrabCut directly
                        self.dataHandler.grabCutOnActiveAnnotations();
                    } else {
                        // no element active (yet); only set action
                        self.setAction(ACTIONS.GRAB_CUT);
                    }
                });
                dtControls.append(grabCutBtn);
            }

            // extra select button here too
            let selectButton_alt = $('<button id="select-button" class="btn btn-sm btn-secondary active" title="Select (S)"><img src="/static/interface/img/controls/select.svg" style="height:18px" /></button>');
            selectButton_alt.on('click', function() {
                self.setAction(ACTIONS.DO_NOTHING);
            });
            dtControls.append(selectButton_alt);
        }

        let removeAnnoCallback = function() {};
        if(window.annotationType !== 'segmentationMasks') {
            // remove button
            removeAnnoCallback = function() {
                // remove active annotations
                var numRemoved = self.dataHandler.removeActiveAnnotations();
                if(numRemoved === 0) {
                    // no active annotation, set action instead
                    self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
                }
            }
            var removeAnnoBtn = $('<button id="remove-annotation" class="btn btn-sm btn-primary" title="Remove Annotation (R)">-</button>');
            removeAnnoBtn.click(removeAnnoCallback);
            this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = removeAnnoBtn;
            dtControls.append(removeAnnoBtn);
        } else {
            removeAnnoCallback = function() {
                self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
            };
        }

        // burst mode checkbox
        var burstModeCallback = function() {
            var chkbx = $('#burst-mode-check');
            self.burstMode = !self.burstMode;
            chkbx.prop('checked', self.burstMode);
            window.setCookie('burstModeEnabled', self.burstMode);
        };
        let burstModeCheck = $('<input type="checkbox" id="burst-mode-check" class="custom-control-input inline-control" style="margin-right:2px" title="Enable burst mode (M)" />');
        burstModeCheck.change(burstModeCallback);
        burstModeCheck.prop('checked', this.burstMode);
        let burstModeCheckContainer = $('<div class="custom-control custom-switch inline-control"></div>');
        burstModeCheckContainer.append(burstModeCheck);
        burstModeCheckContainer.append($('<label for="burst-mode-check" class="custom-control-label inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="Enable burst mode (M)">burst mode</label>'));
        dtControls.append(burstModeCheckContainer);

        if(window.enableEmptyClass) {
            var clearAllCallback = function() {
                self.dataHandler.clearLabelInAll();
            }
            var clearAllBtn = $('<button class="btn btn-sm btn-warning" id="clearAll-button" title="Clear all Annotations (C)">Clear All</button>');
            clearAllBtn.click(clearAllCallback);
            dtControls.append(clearAllBtn);
        }
        
        if(window.annotationType !== 'segmentationMasks') {
            // label all and unsure buttons
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

            // prediction threshold controls
            let predThreshContainer = $('<div class="inline-control" id="prediction-threshold-container"></div>');
            predThreshContainer.append($('<div style="margin:5px">AI</div>'));
            let predThreshRangeContainer = $('<table id="pred-thresh-ranges"></table>');
            let ptrc_vis = $('<tr class="prediction-range-container"></tr>');
            ptrc_vis.append($('<td>Visibility</td>'));
            let predThreshRange_vis = $('<input type="range" id="pred-thresh-vis" min="0" max="100" value="50" />');
            predThreshRange_vis.on({
                'input': function() {
                    let value = 1 - parseFloat($(this).val() / 100.0);
                    if(isNaN(value)) {
                        value = 0.5;
                        $(this).val(value);
                    }
                    window.showPredictions = (value > 0);
                    window.showPredictions_minConf = value;

                    self.dataHandler.setPredictionsVisible(window.showPredictions_minConf);
                },
                'change': function () {
                    let value = parseFloat($(this).val());
                    let cookieVals = window.getCookie('predThreshVis', true);
                    if(cookieVals === null || typeof(cookieVals) !== 'object') {
                        cookieVals = {};
                    }
                    cookieVals[window.projectShortname] = value;
                    window.setCookie('predThreshVis', cookieVals);
                }
            });
            try {
                let value = window.getCookie('predThreshVis', true)[window.projectShortname];
                predThreshRange_vis.val(value);
            } catch {
                predThreshRange_vis.val(100 - window.showPredictions_minConf * 100);
            }
            let rangeTd_vis = $('<td></td>');
            rangeTd_vis.append(predThreshRange_vis);
            ptrc_vis.append(rangeTd_vis);
            predThreshRangeContainer.append(ptrc_vis);
            let ptrc_convert = $('<tr class="prediction-range-container"></tr>');
            ptrc_convert.append($('<td>Conversion</td>'));
            let predThreshRange_convert = $('<input type="range" id="pred-thresh-convert" min="0" max="100" value="95" />');
            predThreshRange_convert.on({
                'input': function() {
                    let value = 1 - parseFloat($(this).val() / 100.0);
                    if(isNaN(value)) {
                        value = 0.05;
                        $(this).val(value);
                    }
                    window.carryOverPredictions = (value > 0);
                    window.carryOverPredictions_minConf = value;

                    // re-convert predictions into annotations
                    self.dataHandler.convertPredictions();
                },
                'change': function() {
                    let value = parseFloat($(this).val());
                    let cookieVals = window.getCookie('predThreshConv', true);
                    if(cookieVals === null || typeof(cookieVals) !== 'object') {
                        cookieVals = {};
                    }
                    cookieVals[window.projectShortname] = value;
                    window.setCookie('predThreshConv', cookieVals);
                }
            });
            try {
                let value = window.getCookie('predThreshConv', true)[window.projectShortname];
                predThreshRange_convert.val(value);
            } catch {
                predThreshRange_convert.val(100 - window.carryOverPredictions_minConf * 100);
            }
            let rangeTd_convert = $('<td></td>');
            rangeTd_convert.append(predThreshRange_convert);
            ptrc_convert.append(rangeTd_convert);
            predThreshRangeContainer.append(ptrc_convert);
            predThreshContainer.append(predThreshRangeContainer);
            dtControls.append(predThreshContainer);
        }

        // semantic segmentation controls
        if(window.annotationType === 'segmentationMasks') {
            this.segmentation_controls = {
                brush: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/spraypaint.svg" style="height:18px" title="Paint" /></button>'),
                brush_rectangle: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/rectangle.svg" style="height:18px" title="Square brush" /></button>'),
                brush_circle: $('<button class="btn btn-sm btn-secondary inline-control"><img src="/static/interface/img/controls/circle.svg" style="height:18px" title="Circular brush" /></button>'),
                brush_size: $('<input class="inline-control" type="number" min="1" max="255" value="20" title="Brush size" style="width:50px" />'),
                erase: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/erase.svg" style="height:18px" title="Erase" /></button>'),
                select_polygon: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/lasso.svg" style="height:18px" title="Select by polygon" /></button>'),
                select_polygon_magnetic: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/lasso_magnetic.svg" style="height:18px" title="Select by magnetic polygon" /></button>'),
                paint_bucket: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/paintbucket.svg" style="height:18px" title="Fill selected area" /></button>'),
                erase_selection: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/erase_selection.svg" style="height:18px" title="Clear selected area" /></button>'),
                clear_selection: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/clear_selection.svg" style="height:18px" title="Clear selection" /></button>'),
                opacity: $('<input class="inline-control" type="range" min="0" max="255" value="220" title="Segmentation opacity" style="width:100px" />')        //TODO: make available for other annotation types as well?
            };

            this.segmentation_controls.brush.on('click', function() {
                // if(getBrushType() === undefined) setBrushType('square');        //TODO
                self.setAction(ACTIONS.ADD_ANNOTATION);
            });
            this.staticButtons[ACTIONS.ADD_ANNOTATION] = this.segmentation_controls.brush;

            this.segmentation_controls.erase.on('click', function() {
                // if(getBrushType() === undefined) setBrushType('square');        //TODO
                self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
            });
            this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = this.segmentation_controls.erase;

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
            this.segmentation_controls.select_polygon.on('click', function() {
                self.setAction(ACTIONS.ADD_SELECT_POLYGON);
            });
            this.segmentation_controls.select_polygon_magnetic.on('click', function() {
                self.setAction(ACTIONS.ADD_SELECT_POLYGON_MAGNETIC);
            });
            this.segmentation_controls.paint_bucket.on('click', function() {
                self.setAction(ACTIONS.PAINT_BUCKET);
            });
            this.segmentation_controls.erase_selection.on('click', function() {
                self.setAction(ACTIONS.ERASE_SELECTION);
            });
            this.segmentation_controls.clear_selection.on('click', function() {
                self.removeAllSelectionElements();
            });

            let segControls = $('<div class="inline-control"></div>');
            segControls.append(this.segmentation_controls.brush);
            segControls.append(this.segmentation_controls.erase);
            segControls.append(this.segmentation_controls.brush_rectangle);
            segControls.append(this.segmentation_controls.brush_circle);
            segControls.append($('<span style="margin-left:10px;margin-right:5px;color:white">Size:</span>'));
            segControls.append(this.segmentation_controls.brush_size);
            segControls.append($('<span style="margin-left:5px;color:white">px</span>'));

            segControls.append(this.segmentation_controls.select_polygon);
            segControls.append(this.segmentation_controls.select_polygon_magnetic);

            // GrabCut
            let grabCutBtn = $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/grabcut.svg" style="height:18px" title="Grab Cut" /></button>');
            grabCutBtn.on('click', function() {
                self.setAction(ACTIONS.GRAB_CUT);
            });
            segControls.append(grabCutBtn);

            segControls.append(this.segmentation_controls.paint_bucket);
            segControls.append(this.segmentation_controls.erase_selection);
            segControls.append(this.segmentation_controls.clear_selection);

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
            if(window.uiBlocked || window.shortcutsDisabled) return;
            
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
                    removeAnnoCallback();

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
            if(window.uiBlocked || window.shortcutsDisabled) return;
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
                        skipEmpty: $('#review-skip-empty').prop('checked'),
                        goldenQuestionsOnly: $('#review-golden-questions-only').prop('checked')
                    }),
                    success: function(data) {
                        if(data.hasOwnProperty('error')) {
                            // e.g. no annotations made yet
                            slider.prop('disabled', true);

                        } else {
                            slider.prop('disabled', false);

                            if(data.hasOwnProperty('minTimestamp')) {
                                slider.prop('min', parseInt(data['minTimestamp'])-1);
                                slider.val(parseInt(data['minTimestamp'])-1);
                                dateSpan.html(new Date(data['minTimestamp']*1000 - 1).toLocaleString());
                            }
                            if(data.hasOwnProperty('maxTimestamp')) {
                                slider.prop('max', parseInt(data['maxTimestamp'])+1);
                            }
                        }
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
            $('#review-golden-questions-only').change(onChange);

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
            if(window.isAdmin) {        //if(window.getCookie('isAdmin') === 'y') {
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
                            $('#review-controls').append($('<div style="color:gray;font-size:9pt;font-style:italic;white-space:normal;word-wrap:break-word;">If no user is selected, only your own annotations are shown.</div>'));

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

    removeAllSelectionElements() {
        this.dataHandler.removeAllSelectionElements();
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

    setLoupeMagnification(factor) {
        if(typeof(factor) === 'number') {
            window.loupeMagnification = Math.max(0.01, Math.min(0.15, factor));
        }
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