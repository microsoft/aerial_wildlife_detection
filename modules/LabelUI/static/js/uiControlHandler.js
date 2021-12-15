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
    GRAB_CUT: 11,
    MAGIC_WAND: 12,
    SELECT_SIMILAR: 13,
    SIMPLIFY_POLYGON: 14
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



class ButtonGroup {
    /**
     * Maintains a dict of Bootstrap buttons and changes their type
     * ("btn-primary", "btn-secondary") depending on which one is active.
     */
    constructor(activeClass, defaultClass) {
        this.buttons = {};
        this.activeClass = (typeof(activeClass) === 'string' ? activeClass : 'btn-primary');
        this.defaultClass = (typeof(defaultClass) === 'string' ? defaultClass : 'btn-secondary');
    }

    set_active_button(id) {
        if(typeof(id) === 'object') {
            // DOM element provided; extract id
            id = $(id).attr('group-id');
        }
        for(var key in this.buttons) {
            if(key == id) {
                this.buttons[key].removeClass(this.defaultClass);
                this.buttons[key].addClass(this.activeClass);
            } else {
                this.buttons[key].removeClass(this.activeClass);
                this.buttons[key].addClass(this.defaultClass);
            }
        }
    }

    addButton(button, callback, idOverride) {
        button = $(button);
        let id = idOverride;
        if(id === undefined) {
            let bID = button.attr('id');
            id = (bID !== undefined ? bID : button.html().toString());
        }
        button.attr('group-id', id);
        button.removeClass(this.activeClass);
        button.addClass(this.defaultClass);
        let self = this;
        if(this.buttons.hasOwnProperty(id)) {
            // button already exists; update callback
            this.buttons[id].off('click');
        } else {
            // add new
            this.buttons[id] = button;
        }
        this.buttons[id].on('click', function(event) {
            self.set_active_button($(this).attr('group-id'));
            if(typeof(callback) === 'function') {
                callback(event);
            }
        });
    }

    removeButton(id) {
        if(this.buttons.hasOwnProperty(id)) {
            this.buttons[id].off('click');
            this.buttons[id].remove();
        }
    }
}


function _set_action_fun(action) {
    return function() {
        this.setAction(action);
    }
}

function _set_brushtype_fun(type) {
    return function() {
        this.setBrushType(type);
    }
}


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

        let self = this;
        this.staticButtons = {};                // buttons that can be pressed in and left that way
        this.toggleButtons = new ButtonGroup(); // buttons that can be switched on or off one at a time

        /*
            viewport controls
        */
        var vpControls = $('#viewport-controls');

        // select
        var selectButton = $('<button id="select-button" class="btn btn-sm btn-secondary active" title="Select (S)"><img src="/static/interface/img/controls/select.svg" style="height:18px" /></button>');
        // this.staticButtons[ACTIONS.DO_NOTHING] = selectButton;
        vpControls.append(selectButton);
        this.toggleButtons.addButton(selectButton, (_set_action_fun(ACTIONS.DO_NOTHING)).bind(this), ACTIONS.DO_NOTHING);
        this.toggleButtons.set_active_button(selectButton);

        // pan
        var panButton = $('<button id="pan-button" class="btn btn-sm btn-secondary" title="Pan (P)"><img src="/static/interface/img/controls/pan.svg" style="height:18px" /></button>');
        // this.staticButtons[ACTIONS.PAN] = panButton;
        vpControls.append(panButton);
        this.toggleButtons.addButton(panButton, (_set_action_fun(ACTIONS.PAN)).bind(this), ACTIONS.PAN);

        // zoom buttons
        let ziButton = $('<button id="zoom-in-button" class="btn btn-sm btn-secondary" title="Zoom In (I)"><img src="/static/interface/img/controls/zoom_in.svg" style="height:18px" /></button>');
        vpControls.append(ziButton);
        this.toggleButtons.addButton(ziButton, (_set_action_fun(ACTIONS.ZOOM_IN)).bind(this), ACTIONS.ZOOM_IN);
        let zoButton = $('<button id="zoom-out-button" class="btn btn-sm btn-secondary" title="Zoom Out (O)"><img src="/static/interface/img/controls/zoom_out.svg" style="height:18px" /></button>');
        vpControls.append(zoButton);
        this.toggleButtons.addButton(zoButton, (_set_action_fun(ACTIONS.ZOOM_OUT)).bind(this), ACTIONS.ZOOM_OUT);

        let zaButton = $('<button id="zoom-area-button" class="btn btn-sm btn-secondary" title="Zoom to Area (Z)"><img src="/static/interface/img/controls/zoom_area.svg" style="height:18px" /></button>');
        // this.staticButtons[ACTIONS.ZOOM_AREA] = zoomAreaButton;
        vpControls.append(zaButton);
        this.toggleButtons.addButton(zaButton, (_set_action_fun(ACTIONS.ZOOM_AREA)).bind(this), ACTIONS.ZOOM_AREA);
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
        let dtControls = $('<div class="data-controls"></div>');

        if(!(['labels', 'segmentationMasks'].includes(window.annotationType))) {
            // add button
            let addAnnoBtn = $('<button id="add-annotation" class="btn btn-sm btn-primary" title="Add Annotation (W)">+</button>');
            this.toggleButtons.addButton(addAnnoBtn, (_set_action_fun(ACTIONS.ADD_ANNOTATION)).bind(this), ACTIONS.ADD_ANNOTATION);
            // this.staticButtons[ACTIONS.ADD_ANNOTATION] = addAnnoBtn;
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
                let magneticPolygonCheckContainer = $('<div id="magnetic-lasso-control" class="toolset-options-container custom-control custom-switch inline-control"></div>');
                magneticPolygonCheckContainer.append(magneticPolygonCheck);
                magneticPolygonCheckContainer.append($('<label for="magnetic-polygon-check" class="custom-control-label inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="Magnetic Polygon"><img src="/static/interface/img/controls/lasso_magnetic.svg" style="height:18px" title="Magnetic Polygon" /></label>'));
                dtControls.append(magneticPolygonCheckContainer);

                // GrabCut
                let grabCutBtn = $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/grabcut_alt.svg" style="height:18px" title="Grab Cut" /></button>');
                grabCutBtn.on('click', function() {
                    // special routine for Polygons: use GrabCut for active polygons, or else set action
                    if(self.dataHandler.getNumActiveAnnotations()) {
                        // at least one element is active; perform GrabCut directly
                        window.taskMonitor.addTask('grabCut', 'Grab Cut');
                        self.dataHandler.grabCutOnActiveAnnotations().then(function() {
                            window.taskMonitor.removeTask('grabCut');
                        });
                    } else {
                        // no element active (yet); only set action
                        self.setAction(ACTIONS.GRAB_CUT);
                    }
                });
                this.toggleButtons.addButton(grabCutBtn, (_set_action_fun(ACTIONS.GRAB_CUT)).bind(this), ACTIONS.GRAB_CUT);
                dtControls.append(grabCutBtn);

                // polygon simplification
                let simplifyBtn = $('<button class="btn btn-sm btn-secondary inline-control active" style="margin-right:2px">Simplify</button>');
                this.toggleButtons.addButton(simplifyBtn, (_set_action_fun(ACTIONS.SIMPLIFY_POLYGON)).bind(this), ACTIONS.SIMPLIFY_POLYGON);
                dtControls.append(simplifyBtn);
                let simplifyControlsCont = $('<div id="simplify-polygon-control" class="toolset-options-container inline-control"></div>');
                let simplifyToleranceSlider = $('<input type="range" id="simplify-polygon-tolerance" min="0" max="100" value="50" style="width:80px" />');
                window.polygonSimplificationTolerance = 0.01;
                simplifyToleranceSlider.on('input', function() {
                    let val = 100.0 - parseFloat($(this).val());
                    window.polygonSimplificationTolerance = .5 / val;
                });
                simplifyControlsCont.append($('<label for="simplify-polygon-tolerance" style="font-size:11pt;margin-top:5px">strength:</label>'));
                simplifyControlsCont.append(simplifyToleranceSlider);
                dtControls.append(simplifyControlsCont);
            }

            // extra select button here too
            let selectButton_alt = $('<button id="select-button" class="btn btn-sm btn-secondary active" title="Select (S)"><img src="/static/interface/img/controls/select.svg" style="height:18px" /></button>');
            this.toggleButtons.addButton(selectButton_alt, (_set_action_fun(ACTIONS.DO_NOTHING)).bind(this));       //TODO: handle multiple buttons with same ID in button group
            dtControls.append(selectButton_alt);
        }

        if(window.annotationType !== 'segmentationMasks') {
            // remove button
            let removeAnnoCallback = function() {
                // remove active annotations
                var numRemoved = self.dataHandler.removeActiveAnnotations();
                if(numRemoved === 0) {
                    // no active annotation, set action instead
                    self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
                }
            }
            var removeAnnoBtn = $('<button id="remove-annotation" class="btn btn-sm btn-primary" title="Remove Annotation (R)">-</button>');
            removeAnnoBtn.click(removeAnnoCallback);
            // this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = removeAnnoBtn;
            this.toggleButtons.addButton(removeAnnoBtn, null, ACTIONS.REMOVE_ANNOTATIONS);
            dtControls.append(removeAnnoBtn);
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
        let burstModeWrapper = $('<div class="toolset-options-container"></div>');
        burstModeWrapper.append(burstModeCheckContainer);
        dtControls.append(burstModeWrapper);

        if(window.enableEmptyClass) {
            let clearAllCallback = function() {
                let callbackYes = function() {
                    window.setUIblocked(false);
                    self.dataHandler.clearLabelInAll();
                }
                window.showYesNoOverlay($('<div>Are you sure you would like to clear all annotations?</div>'), callbackYes, null, 'Yes', 'Cancel');
            }
            let clearAllBtn = $('<button class="btn btn-sm btn-warning" id="clearAll-button" title="Clear all Annotations (C)">Clear All</button>');
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
                brush_rectangle: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/rectangle.svg" style="height:18px" title="square brush" /></button>'),
                brush_circle: $('<button class="btn btn-sm btn-secondary inline-control"><img src="/static/interface/img/controls/circle.svg" style="height:18px" title="circular brush" /></button>'),
                brush_diamond: $('<button class="btn btn-sm btn-secondary inline-control"><img src="/static/interface/img/controls/diamond.svg" style="height:18px" title="diamond brush" /></button>'),
                brush_size: $('<input class="inline-control" type="number" id="brush-size" min="1" max="255" value="20" title="Brush size" style="width:50px" />'),
                erase: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/erase.svg" style="height:18px" title="Erase" /></button>'),
                select_polygon: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/lasso.svg" style="height:18px" title="Select by polygon" /></button>'),
                select_polygon_magnetic: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/lasso_magnetic.svg" style="height:18px" title="Select by magnetic polygon" /></button>'),
                magic_wand: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/magic_wand.svg" style="height:18px" title="magic wand" /></button>'),
                select_similar: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/select_similar.svg" style="height:18px" title="select similar" /></button>'),
                paint_bucket: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/paintbucket.svg" style="height:18px" title="Fill selected area" /></button>'),
                erase_selection: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/erase_selection.svg" style="height:18px" title="Clear selected area" /></button>'),
                clear_selection: $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/clear_selection.svg" style="height:18px" title="Clear selection" /></button>'),
                opacity: $('<input class="inline-control" type="range" min="0" max="255" value="220" title="Segmentation opacity" style="width:100px" />')        //TODO: make available for other annotation types as well?
            };

            // this.segmentation_controls.brush.on('click', function() {
            //     // if(getBrushType() === undefined) setBrushType('square');        //TODO
            //     self.setAction(ACTIONS.ADD_ANNOTATION);
            // });
            this.toggleButtons.addButton(this.segmentation_controls.brush, (_set_action_fun(ACTIONS.ADD_ANNOTATION)).bind(this), ACTIONS.ADD_ANNOTATION);
            // this.staticButtons[ACTIONS.ADD_ANNOTATION] = this.segmentation_controls.brush;

            // this.segmentation_controls.erase.on('click', function() {
            //     // if(getBrushType() === undefined) setBrushType('square');        //TODO
            //     self.setAction(ACTIONS.REMOVE_ANNOTATIONS);
            // });
            // this.staticButtons[ACTIONS.REMOVE_ANNOTATIONS] = this.segmentation_controls.erase;
            this.toggleButtons.addButton(this.segmentation_controls.erase, (_set_action_fun(ACTIONS.REMOVE_ANNOTATIONS)).bind(this), ACTIONS.REMOVE_ANNOTATIONS);

            // brush type
            let brushTypeGroup = new ButtonGroup();
            brushTypeGroup.addButton(this.segmentation_controls.brush_rectangle, (_set_brushtype_fun('rectangle')).bind(this), 'rectangle');
            brushTypeGroup.addButton(this.segmentation_controls.brush_circle, (_set_brushtype_fun('circle')).bind(this), 'circle');
            brushTypeGroup.addButton(this.segmentation_controls.brush_diamond, (_set_brushtype_fun('diamond')).bind(this), 'diamond');
            brushTypeGroup.set_active_button('rectangle');
            // this.segmentation_controls.brush_rectangle.click(function() {
            //     self.setBrushType('rectangle');
            // });
            // this.segmentation_controls.brush_circle.click(function() {
            //     self.setBrushType('circle');
            // });
            // this.segmentation_controls.brush_diamond.click(function() {
            //     self.setBrushType('diamond');
            // });

            // brush size
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

            // area fill tools
            // this.segmentation_controls.select_polygon.on('click', function() {
            //     self.setAction(ACTIONS.ADD_SELECT_POLYGON);
            // });
            // this.segmentation_controls.select_polygon_magnetic.on('click', function() {
            //     self.setAction(ACTIONS.ADD_SELECT_POLYGON_MAGNETIC);
            // });
            // this.segmentation_controls.magic_wand.on('click', function() {
            //     self.setAction(ACTIONS.MAGIC_WAND);
            // });
            // this.segmentation_controls.select_similar.on('click', function() {
            //     self.setAction(ACTIONS.SELECT_SIMILAR);
            // });
            // this.segmentation_controls.paint_bucket.on('click', function() {
            //     self.setAction(ACTIONS.PAINT_BUCKET);
            // });
            // this.segmentation_controls.erase_selection.on('click', function() {
            //     self.setAction(ACTIONS.ERASE_SELECTION);
            // });
            
            this.toggleButtons.addButton(this.segmentation_controls.select_polygon, (_set_action_fun(ACTIONS.ADD_SELECT_POLYGON)).bind(this), ACTIONS.ADD_SELECT_POLYGON);
            this.toggleButtons.addButton(this.segmentation_controls.select_polygon_magnetic, (_set_action_fun(ACTIONS.ADD_SELECT_POLYGON_MAGNETIC)).bind(this), ACTIONS.ADD_SELECT_POLYGON_MAGNETIC);
            this.toggleButtons.addButton(this.segmentation_controls.magic_wand, (_set_action_fun(ACTIONS.MAGIC_WAND)).bind(this), ACTIONS.MAGIC_WAND);
            this.toggleButtons.addButton(this.segmentation_controls.select_similar, (_set_action_fun(ACTIONS.SELECT_SIMILAR)).bind(this), ACTIONS.SELECT_SIMILAR);
            this.toggleButtons.addButton(this.segmentation_controls.paint_bucket, (_set_action_fun(ACTIONS.PAINT_BUCKET)).bind(this), ACTIONS.PAINT_BUCKET);
            this.toggleButtons.addButton(this.segmentation_controls.erase_selection, (_set_action_fun(ACTIONS.ERASE_SELECTION)).bind(this), ACTIONS.ERASE_SELECTION);

            this.segmentation_controls.clear_selection.on('click', function() {
                self.removeAllSelectionElements();
            });

            let segControls = $('<div class="inline-control"></div>');
            segControls.append(this.segmentation_controls.brush);
            segControls.append(this.segmentation_controls.erase);
            let brushOptionsContainer = $('<div class="toolset-options-container inline-control"></div>');
            brushOptionsContainer.append(this.segmentation_controls.brush_rectangle);
            brushOptionsContainer.append(this.segmentation_controls.brush_circle);
            brushOptionsContainer.append(this.segmentation_controls.brush_diamond);
            brushOptionsContainer.append($('<span style="margin-left:10px;margin-right:5px;color:white">Size:</span>'));
            brushOptionsContainer.append(this.segmentation_controls.brush_size);
            brushOptionsContainer.append($('<span style="margin-left:5px;color:white">px</span>'));

            let segIgnoreLabeledContainer = $('<div class="custom-control custom-switch inline-control"></div>');
            let segIgnoreLabeledCheck = $('<input type="checkbox" id="seg-ignore-labeled-check" class="custom-control-input inline-control" style="margin-right:2px" title="preserve labeled pixels" />');
            segIgnoreLabeledCheck.on('change', function() {
                let chckbx = $(this);
                window.segmentIgnoreLabeled = !window.segmentIgnoreLabeled;
                chckbx.prop('checked', window.segmentIgnoreLabeled);
            });
            window.segmentIgnoreLabeled = false;
            segIgnoreLabeledContainer.append(segIgnoreLabeledCheck);
            segIgnoreLabeledContainer.append($('<label for="seg-ignore-labeled-check" class="custom-control-label inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="preserve labeled pixels">keep labeled</label>'));
            let segIgnoreLabeledWrapper = $('<div class="toolset-options-container"></div>');
            segIgnoreLabeledWrapper.append(segIgnoreLabeledContainer);
            brushOptionsContainer.append(segIgnoreLabeledWrapper);
            segControls.append(brushOptionsContainer);

            segControls.append(this.segmentation_controls.select_polygon);
            segControls.append(this.segmentation_controls.select_polygon_magnetic);

            // Magic wand
            segControls.append(this.segmentation_controls.magic_wand);
            let mwToolsContainer = $('<div class="toolset-options-container inline-control"></div>');
            let mwTolerance = $('<input type="number" id="magic-wand-tolerance" min="1" max="255" value="32" />');
            mwTolerance.on('input', function() {
                window.magicWandTolerance = parseFloat($(this).val());
            });
            window.magicWandTolerance = 32.0;
            mwToolsContainer.append($('<label for="magic-wand-tolerance">Tolerance:</label>'));
            mwToolsContainer.append(mwTolerance);
            let mwRadius = $('<input type="number" id="magic-wand-radius" min="1" max="8192" value="120" />');
            mwRadius.on('input', function() {
                window.magicWandRadius = parseInt($(this).val());
            });
            window.magicWandRadius = 120;
            mwToolsContainer.append($('<label for="magic-wand-radius">Max radius:</label>'));
            mwToolsContainer.append(mwRadius);
            segControls.append(mwToolsContainer);

            // select similar
            segControls.append(this.segmentation_controls.select_similar);

            // GrabCut
            let grabCutBtn = $('<button class="btn btn-sm btn-secondary inline-control active"><img src="/static/interface/img/controls/grabcut_alt.svg" style="height:18px" title="Grab Cut" /></button>');
            grabCutBtn.on('click', function() {
                self.setAction(ACTIONS.GRAB_CUT);
            });
            segControls.append(grabCutBtn);

            segControls.append(this.segmentation_controls.paint_bucket);
            segControls.append(this.segmentation_controls.erase_selection);
            
            // toggle to paint/erase all selection areas at once or individually
            let paintAllCallback = function() {
                var chkbx = $('#paint-all-check');
                window.paintbucket_paint_all = chkbx.prop('checked');
                chkbx.prop('checked', window.paintbucket_paint_all);
                // window.setCookie('paintbucket_paint_all', window.paintbucket_paint_all);
            };
            let paintAllCheck = $('<input type="checkbox" id="paint-all-check" class="custom-control-input inline-control" style="margin-right:2px" title="paint/clear selected or all areas at once" />');
            paintAllCheck.change(paintAllCallback);
            window.paintbucket_paint_all = false;
            let paintAllCheckContainer = $('<div id="paint-all-control" class="custom-control custom-switch inline-control"></div>');
            paintAllCheckContainer.append(paintAllCheck);
            paintAllCheckContainer.append($('<label for="paint-all-check" class="custom-control-label inline-control" style="margin-left:0px;margin-right:10px;color:white;cursor:pointer;" title="paint/clear selected or all areas at once">all selections</label>'));
            let paintAllCheckWrapper = $('<div class="toolset-options-container"></div>');
            paintAllCheckWrapper.append(paintAllCheckContainer);
            segControls.append(paintAllCheckWrapper);

            segControls.append(this.segmentation_controls.clear_selection);

            let opacityContainer = $('<div class="toolset-options-container inline-control"></div>');
            opacityContainer.append($('<span style="margin-left:10px;margin-right:5px;color:white">Opacity:</span>'));
            opacityContainer.append(this.segmentation_controls.opacity);
            segControls.append(opacityContainer);

            dtControls.append(segControls);
        }

        // next and previous batch buttons
        var nextBatchCallback = function() {
            self.dataHandler.nextBatch();
        }
        var prevBatchCallback = function() {
            self.dataHandler.previousBatch();
        }

        let interfaceControls = $('#interface-controls');
        
        let prevBatchBtn = $('<button id="previous-button" class="btn btn-sm btn-primary float-left">Previous</button>');
        prevBatchBtn.click(prevBatchCallback);
        interfaceControls.append(prevBatchBtn);
        
        interfaceControls.append(dtControls);
        
        let nextBatchBtn = $('<button id="next-button" class="btn btn-sm btn-primary float-right">Next</button>');
        nextBatchBtn.click(nextBatchCallback);
        interfaceControls.append(nextBatchBtn);



        /*
            Key stroke listener
        */
        $(window).keyup(function(event) {
            if(window.uiBlocked || window.shortcutsDisabled || window.fieldInFocus()) return;
            
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
                    let callbackYes = function() {
                        window.setUIblocked(false);
                        self.dataHandler.clearLabelInAll();
                    }
                    window.showYesNoOverlay($('<div>Are you sure you would like to clear all annotations?</div>'), callbackYes, null, 'Yes', 'Cancel');

                } else if(ccode === 'D' && window.annotationType === 'segmentationMasks') {
                    self.removeAllSelectionElements();

                } else if(ccode === 'E') {
                    self.resetZoom();

                } else if(ccode === 'F') {
                    $('#labelclass-search-box').focus();

                } else if(ccode === 'I') {
                    self.setAction(ACTIONS.ZOOM_IN);
                
                } else if(ccode === 'K' && window.annotationType === 'segmentationMasks') {
                    $('#seg-ignore-labeled-check').trigger('change');

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

        // add mouse wheel listener for input fields
        $(window).on('wheel', function(event) {
            if(!event.shiftKey) return;
            let delta = Math.sign(event.originalEvent.wheelDelta);
            if(typeof(delta) !== 'number') return;
            let action = window.uiControlHandler.getAction();
            if([ACTIONS.ADD_ANNOTATION, ACTIONS.REMOVE_ANNOTATIONS].includes(action) && $('#brush-size').length) {
                let val = Math.max(1, Math.min(255, parseInt($('#brush-size').val()) + delta));
                $('#brush-size').val(val);
                self.setBrushSize(val);
            } else if(action === ACTIONS.MAGIC_WAND && $('#magic-wand-radius').length) {
                if($('#magic-wand-radius').is(':focus')) {
                    let val = Math.max(0, Math.min(8192, parseInt($('#magic-wand-radius').val()) + delta));
                    $('#magic-wand-radius').val(val);
                    window.magicWandRadius = val;
                } else {
                    let val = Math.max(1, Math.min(255, parseInt($('#magic-wand-tolerance').val()) + delta));
                    $('#magic-wand-tolerance').val(val);
                    window.magicWandTolerance = val;
                }
            }
            self.renderAll();
        });
    }


    getAction() {
        return this.action;
    }

    setAction(action) {
        this.action = action;

        // adjust buttons
        this.toggleButtons.set_active_button(action);
        // if(this.staticButtons.hasOwnProperty(action)) {
        //     for(var key in this.staticButtons) {
        //         this.staticButtons[key].removeClass('active');
        //     }
        //     this.staticButtons[action].addClass('active');
        // }

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
            this.segmentation_controls.brush_diamond.removeClass('active');
        } else if(type === 'circle') {
            this.segmentation_properties.brushType = 'circle';
            this.segmentation_controls.brush_rectangle.removeClass('active');
            this.segmentation_controls.brush_circle.addClass('active');
            this.segmentation_controls.brush_diamond.removeClass('active');
        } else if(type === 'diamond') {
            this.segmentation_properties.brushType = 'diamond';
            this.segmentation_controls.brush_rectangle.removeClass('active');
            this.segmentation_controls.brush_circle.removeClass('active');
            this.segmentation_controls.brush_diamond.addClass('active');
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
            $('#loupe-button').removeClass('button-secondary');
            $('#loupe-button').addClass('button-primary');
        } else {
            $('#loupe-button').removeClass('button-primary');
            $('#loupe-button').addClass('button-secondary');
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