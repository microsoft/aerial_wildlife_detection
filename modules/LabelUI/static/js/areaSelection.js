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
        this.activePolygon = null;

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
        if(this.activePolygon !== null && !this.activePolygon.isClosed()) return;

        let element = undefined;
        let promise = null;
        let self = this;
        if(type === 'polygon') {
            element = new PolygonElement(
                this.dataEntry.id + '_selPolygon_' + this.selectionElements.length.toString(),
                startCoordinates,
                SELECTION_STYLE,
                false,
                1000,
                false
            );
            promise = Promise.resolve(element);
        } else if(type === 'magnetic_polygon') {
            window.taskMonitor.addTask('edgeImage', 'finding edges');
            promise = this.dataEntry.renderer.get_edge_image(true)
            .then((edgeMap) => {
                element = new MagneticPolygonElement(
                    self.dataEntry.id + '_selPolygon_' + self.selectionElements.length.toString(),
                    edgeMap,
                    startCoordinates,
                    SELECTION_STYLE,
                    false,
                    1000,
                    false
                );
                window.taskMonitor.removeTask('edgeImage');
                return element;
            });
        }

        promise.then(() => {
            if(element !== undefined) {
                if(self.activePolygon !== null) {
                    // finish active element first
                    if(self.activePolygon instanceof PolygonElement) {
                        if(self.activePolygon.isCloseable()) {
                            // close it
                            self.activePolygon.closePolygon();
                        } else {
                            // polygon was not complete; remove
                            self.removeSelectionElement(self.activePolygon);
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
            if(this.activePolygon === this.selectionElements[idx]) {
                this._set_active_polygon(null);
            }
            this.selectionElements[idx].setActive(false, this.dataEntry.viewport);
            this.dataEntry.viewport.removeRenderElement(this.selectionElements[idx]);
            this.selectionElements.splice(idx, 1);
        }
    }

    removeAllSelectionElements() {
        for(var s in this.selectionElements) {
            this.selectionElements[s].setActive(false, this.dataEntry.viewport);
            this.dataEntry.viewport.removeRenderElement(this.selectionElements[s]);
        }
        this.selectionElements = [];
        this._set_active_polygon(null);
    }

    _set_active_polygon(polygon) {
        if(this.activePolygon !== null) {
            // deactivate currently active polygon first
            this.activePolygon.setActive(false, this.dataEntry.viewport);
            if(this.activePolygon.isCloseable()) {
                this.activePolygon.closePolygon();
            } else {
                // unfinished active polygon; remove
                let idx = this.selectionElements.indexOf(this.activePolygon);
                if(idx !== -1) {
                    this.selectionElements[idx].setActive(false, this.dataEntry.viewport);
                    this.dataEntry.viewport.removeRenderElement(this.selectionElements[idx]);
                    this.selectionElements.splice(idx, 1);
                }
            }
        }
        this.activePolygon = polygon;
        if(this.activePolygon !== null) {
            this.dataEntry.segMap.setActive(false, this.dataEntry.viewport);    //TODO
            this.activePolygon.setActive(true, this.dataEntry.viewport);
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
        } else {
            viewport.removeCallback(this.id, 'mousedown');
            viewport.removeCallback(this.id, 'mousemove');
            viewport.removeCallback(this.id, 'mouseup');
        }
    }

    _canvas_mouseenter(event) {
        if(window.uiBlocked) return;
        let action = window.uiControlHandler.getAction();
        if(![ACTIONS.ADD_SELECT_POLYGON, ACTIONS.ADD_SELECT_POLYGON_MAGNETIC, ACTIONS.PAINT_BUCKET].includes(action)) {
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
        if(action === ACTIONS.ADD_SELECT_POLYGON) {
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
            this.activePolygon !== null) {
            // polygon was being drawn but isn't anymore
            if(this.activePolygon.isCloseable()) {
                // close it
                this.activePolygon.closePolygon();
            } else {
                // polygon was not complete; remove
                this.dataEntry.viewport.removeRenderElement(this.activePolygon);
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
            // find clicked element(s)
            let numElements = 0;
            for(var s in this.selectionElements) {
                if(this.selectionElements[s].containsPoint(mousePos)) {
                    // fill it... (TODO: we assume the base class has a segMap property anyway)
                    let color = (action === ACTIONS.PAINT_BUCKET ? window.labelClassHandler.getActiveColor() : null);
                    this.dataEntry.segMap.paint_bucket(
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
            if(this.activePolygon !== null && this.activePolygon.isClosed()) {
                // still in selection polygon mode but polygon is closed
                if(!this.activePolygon.isActive) {
                    if(this.activePolygon.containsPoint(mousePos)) {
                        this.activePolygon.setActive(true, this.dataEntry.viewport);
                    } else {
                        this.removeSelectionElement(this.activePolygon);
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
        } else if(action === ACTIONS.GRAB_CUT) {
            // get clicked polygon
            for(var e in this.selectionElements) {
                if(this.selectionElements[e].containsPoint(mousePos)) {
                    // clicked into polygon; apply GrabCut
                    // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
                    window.taskMonitor.addTask('grabCut', 'Grab Cut');
                    let coords_in = this.selectionElements[e].getProperty('coordinates');
                    let self = this;
                    try {
                        this.grabCut(coords_in).then((coords_out) => {
                            if(Array.isArray(coords_out) && Array.isArray(coords_out[0]) && coords_out[0].length >= 6) {
                                self.selectionElements[e].setProperty('coordinates', coords_out[0]);
                            } else {
                                if(typeof(coords_out['message']) === 'string') {
                                    window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+coords_out['message'].toString()+'").', 'error', 0);
                                } else {
                                    window.messager.addMessage('No refinement found by GrabCut.', 'regular');
                                }
                            }
                            window.taskMonitor.removeTask('grabCut');
                        });
                    } catch(error) {
                        window.messager.addMessage('An error occurred trying to run GrabCut on selection (message: "'+error.toString()+'").', 'error', 0);
                        window.taskMonitor.removeTask('grabCut');
                    }
                }
            }
        } else if(action === ACTIONS.MAGIC_WAND) {
            // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
            window.taskMonitor.addTask('magicWand', 'magic wand');
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
                    window.taskMonitor.removeTask('magicWand');
                });
            } catch(error) {
                window.messager.addMessage('An error occurred trying to run magic wand (message: "'+error.toString()+'").', 'error', 0);
                window.taskMonitor.removeTask('magicWand');
            }
        } else if(action === ACTIONS.SELECT_SIMILAR) {
            // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
            window.taskMonitor.addTask('selSim', 'select similar');
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
                        window.taskMonitor.removeTask('selSim');
                    });
                } else {
                    // no polygon clicked; use magic wand first
                    let mwCoords = null;
                    window.taskMonitor.addTask('magicWand_pre', 'magic wand');
                    this.magicWand(this.dataEntry.fileName, mousePos, window.magicWandTolerance, window.magicWandRadius, false).then((coords_out) => {
                        window.taskMonitor.removeTask('magicWand_pre');
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
                            window.taskMonitor.removeTask('selSim');
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
                                window.taskMonitor.removeTask('selSim');
                            });
                        }
                    });
                }
            } catch(error) {
                window.messager.addMessage('An error occurred trying to find similar regions (message: "'+error.toString()+'").', 'error', 0);
                window.taskMonitor.removeTask('selSim');
            }
        }

        if(action !== ACTIONS.MAGIC_WAND) {
            this.dataEntry.viewport.removeRenderElement(this.magicWand_rectangle);  //TODO: store bool for better efficiency?
        }
    }
}