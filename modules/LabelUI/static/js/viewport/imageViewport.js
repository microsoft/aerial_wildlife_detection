/*
    Interface to the HTML canvas for regular image annotations.

    2019 Benjamin Kellenberger
*/

class ImageViewport {

    constructor(canvas) {
        this.canvas = canvas;
        var self = this;
        $(window).on('resize', function() {
            //TODO: doesn't allow for zooming
            // self.setViewport([0, 0, parseInt(self.canvas.width()), parseInt(self.canvas.height())]);
            self.render();
        });
        this.ctx = canvas[0].getContext('2d');
        this.validArea = [0, 0, 1, 1];    // may be a part of the canvas if the main image is smaller
        this.viewport = [0, 0, canvas[0].width, canvas[0].height];
        this.renderStack = [];
        this.renderStack.sortFun = (function(a, b) {
            if(a.zIndex < b.zIndex) {
                return -1;
            } else if(a.zIndex > b.zIndex) {
                return 1;
            } else {
                return 0;
            }
        });
        this._setupCallbacks();
    }

    _setupCallbacks() {
        this.callbacks = {};
        for(var i=0; i<window.eventTypes.length; i++) {
            this.callbacks[window.eventTypes[i]] = {};
        }
        this._updateCallbacks();
    }

    _updateCallbacks() {
        var self = this;
        for(var i=0; i<window.eventTypes.length; i++) {
            var type = window.eventTypes[i];
            if(Object.keys(this.callbacks[type]).length == 0) {
                // disable callback
                $(this.canvas).off(type);
            } else {
                // enable callback
                $(this.canvas).off(type);
                $(this.canvas).on(type, function(event) {
                    for(var key in self.callbacks[event.type]) {
                        self.callbacks[event.type][key](event);
                    }
                    self.render();
                });
            }
        }
    }

    _getCanvasScaleFactors() {
        var scaleX = this.canvas[0].width;
        var scaleY = this.canvas[0].height;
        return [scaleX, scaleY];
    }

    addRenderElement(element) {
        if(this.indexOfRenderElement(element) === -1) {
            this.renderStack.push(element);
            this.renderStack.sort(this.renderStack.sortFun);
            this.render();
        }
    }

    indexOfRenderElement(element) {
        return(this.renderStack.indexOf(element));
    }

    updateRenderElement(index, element) {
        this.renderStack[index] = element;
        this.renderStack.sort(this.renderStack.sortFun);
        this.render();
    }

    removeRenderElement(element) {
        var idx = this.renderStack.indexOf(element);
        if(idx !== -1) {
            this.renderStack.splice(idx, 1);
            this.render();
        }
    }

    getRelativeCoordinates(event, offsetCompensated) {
        var posX = event.pageX - this.canvas.offset().left;
        var posY = event.pageY - this.canvas.offset().top;

        //TODO: ugly hack: need to account for differences between canvas size and canvas DOM element size...
        // var scale = [this.canvas[0].width/this.canvas.width(), this.canvas[0].height/this.canvas.height()];
        var coords = this.transformCoordinates([posX, posY], 'validArea', true);
        // if(offsetCompensated) {
        //     coords = [coords[0] - this.validArea[0], coords[1] - this.validArea[1]];
        // }
        return coords;
    }

    _scale(coordinates, target) {
        var scaleFactors = this._getCanvasScaleFactors();
        if(target !=='canvas') {
            scaleFactors = [1/scaleFactors[0], 1/scaleFactors[1]];
        }
        var coordsOut = [];
        coordsOut.push((coordinates[0]) * scaleFactors[0]);
        coordsOut.push((coordinates[1]) * scaleFactors[1]);

        if(coordinates.length == 4) {
            coordsOut.push(coordinates[2] * scaleFactors[0]);
            coordsOut.push(coordinates[3] * scaleFactors[1]);
        }
        return coordsOut;
    }

    scaleToCanvas(coordinates) {
        return this._scale(coordinates, 'canvas');
    }

    scaleToViewport(coordinates) {
        return this.transformCoordinates(coordinates, 'validArea', true);
        // return this._scale(coordinates, 'viewport');
    }

    transformCoordinates(coordinates, target, backwards) {
        var coords_out = coordinates.slice();
        var canvasSize = [this.canvas[0].width, this.canvas[0].height];
        if(backwards) {
            canvasSize = [1/canvasSize[0], 1/canvasSize[1]];
        }

        if(target === 'canvas') {
            // scale w.r.t. full canvas size
            coords_out[0] *= canvasSize[0];
            coords_out[1] *= canvasSize[1];
            if(coords_out.length == 4) {
                coords_out[2] *= canvasSize[0];
                coords_out[3] *= canvasSize[1];
            }

        } else if(target === 'validArea') {
            // shift and scale w.r.t. valid area
            var canvasScaleRatio = [this.canvas[0].width/this.canvas.width(), this.canvas[0].height/this.canvas.height()];
            var areaSize = [this.canvas[0].width * this.validArea[2], this.canvas[0].height * this.validArea[3]];
            if(backwards) {
                coords_out[0] = coords_out[0] / (areaSize[0] / canvasScaleRatio[0]) - this.validArea[0];
                coords_out[1] = coords_out[1] / (areaSize[1] / canvasScaleRatio[1]) - this.validArea[1];
                if(coords_out.length == 4) {
                    coords_out[2] /= areaSize[0];
                    coords_out[3] /= areaSize[1];
                }
            } else {
                coords_out[0] = (coords_out[0] + this.validArea[0]) * areaSize[0] / canvasScaleRatio[0];
                coords_out[1] = (coords_out[1] + this.validArea[1]) * areaSize[1] / canvasScaleRatio[1];
                if(coords_out.length == 4) {
                    coords_out[2] *= areaSize[0];
                    coords_out[3] *= areaSize[1];
                }
            }
        }
        return coords_out;
    }

    render() {
        // clear canvas
        var extent = [0, 0, this.canvas[0].width, this.canvas[0].height];
        this.ctx.fillStyle = window.styles.background;
        this.ctx.fillRect(0, 0, extent[2], extent[3]);

        // iterate through render stack
        var self = this;
        var scaleFun = function(coords, target) {
            return self.transformCoordinates(coords, target, false);
        }
        for(var i=0; i<this.renderStack.length; i++) {
            this.renderStack[i].render(this.ctx, this.viewport, this.validArea, scaleFun);
        }
    }

    setViewport(viewport) {
        this.viewport = viewport;
        this.render();
    }

    resetViewport() {
        this.viewport = [0, 0, this.canvas.width(), this.canvas.height()];
        this.render();
    }

    setValidArea(area) {
        this.validArea = area;
        this.render();
    }

    addCallback(id, type, callbackFun) {
        if(!(id in this.callbacks[type])) {
            this.callbacks[type][id] = callbackFun;
            this._updateCallbacks();
        }
    }

    removeCallback(id, type) {
        if(id in this.callbacks[type]) {
            delete this.callbacks[type][id];
            this._updateCallbacks();
        }
    }
}