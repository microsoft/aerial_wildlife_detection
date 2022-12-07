/**
 * Tools and routines for area selection (polygons, etc.) in an image, e.g. for
 * segmentation.
 * 
 * 2021 Benjamin Kellenberger
 */

//TODO: query from server
const SELECTION_STYLE = {
    'fillColor': '#0066AA33',
    'strokeColor': '#0066AAFF',
    'lineOpacity': 1.0,
    'lineWidth': 2.0,
    'lineDash': [10],
    'lineDashOffset': 0.0,
    'refresh_rate': 100
    //TODO: make proper animate entry
}


class AreaSelector {

    constructor(dataEntry) {
        this.dataEntry = dataEntry;
        this.id = this.dataEntry.id + '_areaSel';
        this.selectionElements = [];
        this.activeSelectionElement = null;

        this._set_listeners_active(true);   //TODO

        // init edge image for faster magnetic lasso activation
        this.dataEntry.renderer.get_edge_image(true);

        // rectangle that previews the spatial range of the magic wand
        this.magicWand_rectangle = new PaintbrushElement(
            'magicWand_preview_rect',
            null, null, 1000,
            true,
            2*window.magicWandRadius,
            'rectangle'
        );
    }

    convexHull(polygon) {
        /**
         * Receives coordinates and returns their convex hull.
         */
        if(!Array.isArray(polygon) || polygon.length < 6) return undefined;
        if(polygon.length === 6) return polygon;
        let points = [];
        for(var p=0; p<polygon.length; p+=2) {
            points.push({x: polygon[p], y: polygon[p+1]});
        }
        let cHull = convexhull.makeHull(points);
        let polygon_out = [];
        for(var c=0; c<cHull.length; c++) {
            polygon_out.push(cHull[c].x);
            polygon_out.push(cHull[c].y);
        }
        return polygon_out;
    }

    grabCut(polygon) {
        /**
         * Runs GrabCut for a given data entry and a given polygon.
         */
        return this.dataEntry.grabCut(polygon);
    }

    magicWand(fileName, mousePos, tolerance, maxRadius, rgbOnly) {
        /**
         * Performs a magic wand operation on the image.
         */
        return $.ajax({
            url: window.baseURL + 'magic_wand',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({
                image_path: fileName,
                seed_coordinates: mousePos,
                tolerance: tolerance,
                max_radius: maxRadius,
                rgb_only: rgbOnly
            })
        })
        .then((data) => {
            if(data.hasOwnProperty('result')) {
                return data['result'];       // Array of coordinates
            } else {
                return data;
            }
        });
    }

    selectSimilar(fileName, seedPolygon, tolerance, numMax) {
        /**
         * Finds similar regions in the image based on a seed polygon and
         * returns their polygons accordingly.
         */
        return $.ajax({
            url: window.baseURL + 'select_similar',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({
                image_path: fileName,
                seed_polygon: seedPolygon,
                tolerance: tolerance,
                num_max: numMax,
                return_polygon: true
            })
        })
        .then((data) => {
            if(data.hasOwnProperty('result')) {
                return data['result'];       // Array of coordinates
            } else {
                return data;
            }
        });
    }

    addSelectionElement(type, startCoordinates) {
        /**
         * Adds a new selection element, unless there is a current one that is
         * not yet closed.
         */
        if(this.activeSelectionElement !== null && (this.activeSelectionElement instanceof PolygonElement && !this.activeSelectionElement.isClosed())) return;

        let element = undefined;
        let promise = null;
        let self = this;
        if(type === 'rectangle') {
            element = new RectangleElement(
                this.dataEntry.id + '_selRect_' + (new Date()).toString(),
                startCoordinates[0], startCoordinates[1],
                0.0001, 0.0001,
                SELECTION_STYLE,
                false,
                1000
            );
            element.activeHandle = 'se';
            promise = Promise.resolve(element);
        } else if(type === 'polygon') {
            element = new PolygonElement(
                this.dataEntry.id + '_selPolygon_' + (new Date()).toString(),
                startCoordinates,
                SELECTION_STYLE,
                false,
                1000
            );
            promise = Promise.resolve(element);
        } else if(type === 'magnetic_polygon') {
            window.jobIndicator.addJob('edgeImage', 'finding edges');
            promise = this.dataEntry.renderer.get_edge_image(true)
            .then((edgeMap) => {
                element = new MagneticPolygonElement(
                    self.dataEntry.id + '_selPolygon_' + self.selectionElements.length.toString(),
                    edgeMap,
                    startCoordinates,
                    SELECTION_STYLE,
                    false,
                    1000
                );
                window.jobIndicator.removeJob('edgeImage');
                return element;
            });
        }

        promise.then(() => {
            if(element !== undefined) {
                if(self.activeSelectionElement !== null) {
                    // finish active element first
                    if(self.activeSelectionElement instanceof PolygonElement) {
                        if(self.activeSelectionElement.isCloseable()) {
                            // close it
                            self.activeSelectionElement.closePolygon();
                        } else {
                            // polygon was not complete; remove
                            self.removeSelectionElement(self.activeSelectionElement);
                        }
                    }
                    self._set_active_polygon(null);
                }
    
                self.dataEntry.viewport.addRenderElement(element);
                element.setActive(true, self.dataEntry.viewport);
                element.mouseDrag = true;
                self.dataEntry.segMap.setActive(false, self.dataEntry.viewport);    //TODO
                // window.uiControlHandler.setAction(ACTIONS.ADD_ANNOTATION);      // required for polygon drawing (TODO: conflicts with burst mode)
                self.selectionElements.push(element);
                self._set_active_polygon(element);
            }
        });
    }

    removeSelectionElement(element) {
        let idx = this.selectionElements.indexOf(element);
        if(idx !== -1) {
            if(this.activeSelectionElement === this.selectionElements[idx]) {
                this._set_active_polygon(null);
            }
            if(typeof(this.selectionElements[idx]) === 'object') this.selectionElements[idx].setActive(false, this.dataEntry.viewport);
            this.dataEntry.viewport.removeRenderElement(this.selectionElements[idx]);
            this.selectionElements.splice(idx, 1);
        }
    }

    removeActiveSelectionElements() {
        for(var s=this.selectionElements.length-1; s>=0; s--) {
            if(this.selectionElements[s].isActive) {
                if(typeof(this.selectionElements[s]) === 'object') this.selectionElements[s].setActive(false, this.dataEntry.viewport);
                this.dataEntry.viewport.removeRenderElement(this.selectionElements[s]);
            }
        }
        this._set_active_polygon(null);
    }

    removeAllSelectionElements() {
        for(var s in this.selectionElements) {
            if(typeof(this.selectionElements[s]) === 'object') this.selectionElements[s].setActive(false, this.dataEntry.viewport);
            this.dataEntry.viewport.removeRenderElement(this.selectionElements[s]);
        }
        this.selectionElements = [];
        this._set_active_polygon(null);
    }

    closeActiveSelectionElement() {
        if(this.activeSelectionElement === null || !(this.activeSelectionElement instanceof PolygonElement) || !this.activeSelectionElement.isCloseable()) return;
        this.activeSelectionElement.closePolygon();
    }

    _set_active_polygon(polygon) {
        if(this.activeSelectionElement !== null) {
            // deactivate currently active polygon first
            this.activeSelectionElement.setActive(false, this.dataEntry.viewport);
            if(this.activeSelectionElement instanceof PolygonElement && this.activeSelectionElement.isCloseable()) {
                this.activeSelectionElement.closePolygon();
            } else {
                // unfinished active polygon; remove
                let idx = this.selectionElements.indexOf(this.activeSelectionElement);
                if(idx !== -1) {
                    this.selectionElements[idx].setActive(false, this.dataEntry.viewport);
                    this.dataEntry.viewport.removeRenderElement(this.selectionElements[idx]);
                    this.selectionElements.splice(idx, 1);
                }
            }
        }
        this.activeSelectionElement = polygon;
        if(this.activeSelectionElement !== null) {
            this.dataEntry.segMap.setActive(false, this.dataEntry.viewport);    //TODO
            this.activeSelectionElement.setActive(true, this.dataEntry.viewport);
        } else {
            this.dataEntry.segMap.setActive(true, this.dataEntry.viewport);    //TODO
        }
    }

    _update_magic_wand_preview_rectangle(event) {
        if(window.magicWandRadius > 0) {
            let mousePos = this.dataEntry.viewport.getRelativeCoordinates(event, 'validArea');
            this.magicWand_rectangle.x = mousePos[0];
            this.magicWand_rectangle.y = mousePos[1];
            this.magicWand_rectangle.size = 2*window.magicWandRadius;
        } else {
            // no restriction in area; hide rectangle
            this.magicWand_rectangle.size = null;
        }
    }

    _set_listeners_active(active) {
        let viewport = this.dataEntry.viewport;
        let self = this;
        if(active) {
            viewport.addCallback(this.id, 'mouseenter', function(event) {
                self._canvas_mouseenter(event);
            });
            viewport.addCallback(this.id, 'mousedown', function(event) {
                self._canvas_mousedown(event);
            });
            viewport.addCallback(this.id, 'mousemove', function(event) {
                self._canvas_mousemove(event);
            });
            viewport.addCallback(this.id, 'mouseup', function(event) {
                self._canvas_mouseup(event);
            });
            viewport.addCallback(this.id, 'keyup', function(event) {
                self._canvas_keyup(event);
            });
        } else {
            viewport.removeCallback(this.id, 'mousedown');
            viewport.removeCallback(this.id, 'mousemove');
            viewport.removeCallback(this.id, 'mouseup');
            viewport.removeCallback(this.id, 'keyup');
        }
    }

    _canvas_mouseenter(event) {
        if(window.uiBlocked) return;
        let action = window.uiControlHandler.getAction();
        if(![ACTIONS.ADD_SELECT_RECTANGLE, ACTIONS.ADD_SELECT_POLYGON, ACTIONS.ADD_SELECT_POLYGON_MAGNETIC, ACTIONS.PAINT_BUCKET].includes(action) &&
            (this.activeSelectionElement === null || !this.activeSelectionElement.isActive)) {
            // user changed action (e.g. to paint function); cancel active polygon and set segmentation map active instead
            this._set_active_polygon(null);
        }
        if(action === ACTIONS.MAGIC_WAND) {
            // add preview rectangle
            this._update_magic_wand_preview_rectangle(event);
            this.dataEntry.viewport.addRenderElement(this.magicWand_rectangle);
        } else {
            this.dataEntry.viewport.removeRenderElement(this.magicWand_rectangle);  //TODO: store bool for better efficiency?
        }
    }

    _canvas_mousedown(event) {
        if(window.uiBlocked) return;
        let action = window.uiControlHandler.getAction();
        if(action === ACTIONS.ADD_SELECT_RECTANGLE && 
            (this.activeSelectionElement === null || !this.activeSelectionElement.isActive)) {
            let mousePos = this.dataEntry.viewport.getRelativeCoordinates(event, 'validArea');
            this.addSelectionElement('rectangle', mousePos);
        } else if(action === ACTIONS.ADD_SELECT_POLYGON) {
            let mousePos = this.dataEntry.viewport.getRelativeCoordinates(event, 'validArea');
            this.addSelectionElement('polygon', mousePos);
        } else if(action === ACTIONS.ADD_SELECT_POLYGON_MAGNETIC) {
            let mousePos = this.dataEntry.viewport.getRelativeCoordinates(event, 'validArea');
            this.addSelectionElement('magnetic_polygon', mousePos);
        }
    }

    _canvas_mousemove(event) {
        if(window.uiBlocked) return;
        let action = window.uiControlHandler.getAction();
        if(action === ACTIONS.DO_NOTHING &&
            this.activeSelectionElement !== null) {
            // polygon was being drawn but isn't anymore
            if(this.activeSelectionElement instanceof PolygonElement && this.activeSelectionElement.isCloseable()) {
                // close it
                this.activeSelectionElement.closePolygon();
            } else {
                // polygon was not complete; remove
                this.dataEntry.viewport.removeRenderElement(this.activeSelectionElement);
                this._set_active_polygon(null);
            }
        } else if(action === ACTIONS.MAGIC_WAND) {
            this._update_magic_wand_preview_rectangle(event);
        }
    }
    
    _canvas_mouseup(event) {
        if(window.uiBlocked) return;
        let mousePos = this.dataEntry.viewport.getRelativeCoordinates(event, 'validArea');
        let action = window.uiControlHandler.getAction();
        if([ACTIONS.PAINT_BUCKET, ACTIONS.ERASE_SELECTION].includes(action)) {
            // find valid element(s): either clicked or all if window.paintbucket_paint_all
            let numElements = 0;
            for(var s=this.selectionElements.length-1; s>=0; s--) {
                if(window.paintbucket_paint_all || this.selectionElements[s].containsPoint(mousePos)) {
                    // fill it... (TODO: we assume the base class has a segMap property anyway)
                    let color = (action === ACTIONS.PAINT_BUCKET ? window.labelClassHandler.getActiveColor() : null);
                    this.dataEntry.paint_bucket(
                        this.selectionElements[s].getProperty('coordinates'),
                        color,
                        window.segmentIgnoreLabeled
                    );

                    // ...and remove it
                    this.removeSelectionElement(this.selectionElements[s]);
                    numElements++;
                }
            }
            if(numElements === 0 && action === ACTIONS.PAINT_BUCKET) {
                // clicked in blank with paint bucket; ask to paint all unlabeled pixels
                let className = window.labelClassHandler.getActiveClassName();
                let confMarkup = $('<div>Assign all unlabeled pixels to class "' + className + '"?</div>');
                let self = this;
                let callbackYes = function() {
                    self.dataEntry.fill_unlabeled(window.labelClassHandler.getActiveClassID());
                };
                window.showYesNoOverlay(confMarkup, callbackYes, null, 'Yes', 'Cancel')
            }
        } else if([ACTIONS.DO_NOTHING, ACTIONS.ADD_ANNOTATION].includes(action)) {
            if(this.activeSelectionElement !== null && this.activeSelectionElement instanceof PolygonElement && this.activeSelectionElement.isClosed()) {
                // still in selection polygon mode but polygon is closed
                if(!this.activeSelectionElement.isActive) {
                    if(this.activeSelectionElement.containsPoint(mousePos)) {
                        this.activeSelectionElement.setActive(true, this.dataEntry.viewport);
                    } else {
                        this.removeSelectionElement(this.activeSelectionElement);
                    }
                }
            } else {
                // find clicked element...
                for(var s in this.selectionElements) {
                    if(this.selectionElements[s].containsPoint(mousePos)) {
                        // ...and set active again
                        this._set_active_polygon(this.selectionElements[s])
                    }
                }
            }
        } else if(action === ACTIONS.CONVEX_HULL) {
            // get clicked polygon
            let numClicked = 0;
            for(var e in this.selectionElements) {
                if(this.selectionElements[e] instanceof PolygonElement && this.selectionElements[e].containsPoint(mousePos)) {
                    numClicked++;
                    let cHull = this.convexHull(this.selectionElements[e].getProperty('coordinates'));
                    if(Array.isArray(cHull) && cHull.length >= 6) {
                        this.selectionElements[e].setProperty('coordinates', cHull);
                    }
                }
            }
            if(!numClicked) {
                window.messager.addMessage('Click into a drawn polygon to transform it into its convex hull.', 'info');
            }

        } else if(action === ACTIONS.SIMPLIFY_POLYGON) {
            // get clicked polygon
            let numClicked = 0;
            for(var e in this.selectionElements) {
                if(this.selectionElements[e] instanceof PolygonElement && this.selectionElements[e].containsPoint(mousePos)) {
                    numClicked++;
                    let coords_in = this.selectionElements[e].getProperty('coordinates');
                    let coords_out = simplifyPolygon(coords_in, window.polygonSimplificationTolerance, true);      //TODO: hyperparameters
                    if(Array.isArray(coords_out) && coords_out.length >= 6) {
                        this.selectionElements[e].setProperty('coordinates', coords_out);
                    }
                }
            }
            if(!numClicked) {
                window.messager.addMessage('Click into a drawn polygon to simplify it.', 'info');
            }

        } else if(action === ACTIONS.GRAB_CUT) {
            // get clicked polygon
            let numClicked = 0;
            for(var e in this.selectionElements) {
                if(this.selectionElements[e].containsPoint(mousePos)) {
                    // clicked into selection element; apply GrabCut
                    numClicked++;
                    // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
                    window.jobIndicator.addJob('grabCut', 'Grab Cut');
                    let coords_in = this.selectionElements[e].getProperty('coordinates');
                    let self = this;
                    try {
                        this.grabCut(coords_in).then((coords_out) => {
                            if(Array.isArray(coords_out) && Array.isArray(coords_out[0]) && coords_out[0].length >= 6) {
                                if(self.selectionElements[e] instanceof RectangleElement) {
                                    // convert into polygon first
                                    self.removeSelectionElement(self.selectionElements[e]);
                                    self.addSelectionElement('polygon', coords_out[0]);
                                    
                                } else {
                                    self.selectionElements[e].setProperty('coordinates', coords_out[0]);
                                }
                            } else {
                                if(typeof(coords_out['message']) === 'string') {
                                    window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+coords_out['message'].toString()+'").', 'error', 0);
                                } else {
                                    window.messager.addMessage('No refinement found by GrabCut.', 'regular');
                                }
                            }
                            window.jobIndicator.removeJob('grabCut');
                        });
                    } catch(error) {
                        window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+error.toString()+'").', 'error', 0);
                        window.jobIndicator.removeJob('grabCut');
                    }
                }
            }
            if(!numClicked) {
                // no selection element clicked; apply magic wand first
                window.jobIndicator.addJob('magicWand_gc_pre', 'magic wand');
                let self = this;
                this.magicWand(this.dataEntry.fileName, mousePos, window.magicWandTolerance, window.magicWandRadius, false).then((coords_out) => {
                    window.jobIndicator.removeJob('magicWand_gc_pre');
                    if(Array.isArray(coords_out) && coords_out.length >= 6) {
                        return coords_out;
                    } else {
                        return null;
                    }
                })
                .then((coords_in) => {
                    if(coords_in !== null) {
                        // something found; now apply GrabCut
                        try {
                            self.grabCut(coords_in).then((coords_out) => {
                                if(Array.isArray(coords_out) && Array.isArray(coords_out[0]) && coords_out[0].length >= 6) {
                                    self.addSelectionElement('polygon', coords_out[0]);
                                } else {
                                    if(typeof(coords_out['message']) === 'string') {
                                        window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+coords_out['message'].toString()+'").', 'error', 0);
                                    } else {
                                        window.messager.addMessage('No refinement found by GrabCut.', 'regular');
                                    }
                                }
                                window.jobIndicator.removeJob('grabCut');
                            });
                        } catch(error) {
                            window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+error.toString()+'").', 'error', 0);
                            window.jobIndicator.removeJob('grabCut');
                        }
                    } else {
                        window.messager.addMessage('No refinement found by GrabCut.', 'regular');
                    }
                });
            }
        } else if(action === ACTIONS.MAGIC_WAND) {
            // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
            window.jobIndicator.addJob('magicWand', 'magic wand');
            try {
                let tolerance = window.magicWandTolerance;
                let maxRadius = window.magicWandRadius;
                let rgbOnly = false;
                let self = this;
                this.magicWand(this.dataEntry.fileName, mousePos, tolerance, maxRadius, rgbOnly).then((coords_out) => {
                    if(Array.isArray(coords_out) && coords_out.length >= 6) {
                        // remove any previous overlapping selection polygons first
                        for(var e in self.selectionElements) {
                            if(self.selectionElements[e].containsPoint(mousePos)) {
                                self.removeSelectionElement(self.selectionElements[e]);
                            }
                        }

                        // add new selection polygon
                        self.addSelectionElement('polygon', coords_out);
                    } else {
                        if(coords_out !== null && typeof(coords_out['message']) === 'string') {
                            window.messager.addMessage('An error occurred trying to run magic wand (message: "'+coords_out['message'].toString()+'").', 'error', 0);
                        } else {
                            window.messager.addMessage('No area found with magic wand.', 'regular');
                        }
                    }
                    window.jobIndicator.removeJob('magicWand');
                });
            } catch(error) {
                window.messager.addMessage('An error occurred trying to run magic wand (message: "'+error.toString()+'").', 'error', 0);
                window.jobIndicator.removeJob('magicWand');
            }
        } else if(action === ACTIONS.SELECT_SIMILAR) {
            // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
            window.jobIndicator.addJob('selSim', 'select similar');
            try {
                //TODO: hyperparameters
                let tolerance = .25;     //TODO: quantile
                let numMax = 10; //TODO
                let self = this;

                // find clicked polygon
                let seedPolygon = null;
                for(var e in this.selectionElements) {
                    if(this.selectionElements[e].containsPoint(mousePos)) {
                        seedPolygon = this.selectionElements[e].getProperty('coordinates');
                        break;
                    }
                }
                if(seedPolygon !== null) {
                    this.selectSimilar(this.dataEntry.fileName, seedPolygon, tolerance, numMax).then((coords_similar) => {
                        if(Array.isArray(coords_similar) && coords_similar.length) {
                            for(var c=0; c<coords_similar.length; c++) {
                                self.addSelectionElement('polygon', coords_similar[c]);
                            }
                        }
                        window.jobIndicator.removeJob('selSim');
                    });
                } else {
                    // no polygon clicked; use magic wand first
                    let mwCoords = null;
                    window.jobIndicator.addJob('magicWand_pre', 'magic wand');
                    this.magicWand(this.dataEntry.fileName, mousePos, window.magicWandTolerance, window.magicWandRadius, false).then((coords_out) => {
                        window.jobIndicator.removeJob('magicWand_pre');
                        if(Array.isArray(coords_out) && coords_out.length >= 6) {
                            mwCoords = coords_out;
                            return coords_out;
                        } else {
                            return null;
                        }
                    })
                    .then((seed_polygon) => {
                        if(seed_polygon === null) {
                            // nothing found
                            window.jobIndicator.removeJob('selSim');
                        } else {
                            // something found; now select similar
                            self.selectSimilar(self.dataEntry.fileName, seed_polygon, tolerance, numMax).then((coords_similar) => {
                                // add original magic wand polygon...
                                self.addSelectionElement('polygon', mwCoords);

                                // ...and similar others
                                if(Array.isArray(coords_similar) && coords_similar.length) {
                                    for(var c=0; c<coords_similar.length; c++) {
                                        self.addSelectionElement('polygon', coords_similar[c]);
                                    }
                                }
                                window.jobIndicator.removeJob('selSim');
                            });
                        }
                    });
                }
            } catch(error) {
                window.messager.addMessage('An error occurred trying to find similar regions (message: "'+error.toString()+'").', 'error', 0);
                window.jobIndicator.removeJob('selSim');
            }
        }

        if(action !== ACTIONS.MAGIC_WAND) {
            this.dataEntry.viewport.removeRenderElement(this.magicWand_rectangle);  //TODO: store bool for better efficiency?
        }
    }

    _canvas_keyup(event) {
        let ccode = String.fromCharCode(event.which);
        console.log(ccode)
    }
}