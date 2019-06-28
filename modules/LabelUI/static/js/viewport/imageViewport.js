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

    getRelativeCoordinates(event, target) {
        var posX = event.pageX - this.canvas.offset().left;
        var posY = event.pageY - this.canvas.offset().top;
        var coords = this.transformCoordinates([posX, posY], target, true);
        return coords;
    }

    transformCoordinates(coordinates, target, backwards) {
        /*
            Modifies coordinates w.r.t. either the valid area (typically spanned by the image)
            or the full canvas and transforms relative values (in [0, 1]) forward to full, ab-
            solute values, or else full values back to relative scores if "backwards" is true.

            Note that the forward case uses the hypothetical canvas dimensions, whereas the
            "backwards" case relies on the actual canvas DOM size. This is to account for re-
            lative scaling and mouse/touch gesture capturing.
        */
        var coords_out = coordinates.slice();
        
        if(backwards) {
            var canvasSize = [this.canvas.width(), this.canvas.height()];
            if(target === 'canvas') {
                coords_out[0] /= canvasSize[0];
                coords_out[1] /= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] /= canvasSize[0];
                    coords_out[3] /= canvasSize[1];
                }

            } else if(target === 'validArea') {
                var validSize = [this.validArea[2]*canvasSize[0], this.validArea[3]*canvasSize[1]];
                coords_out[0] = (coords_out[0] - (this.validArea[0]*canvasSize[0])) / validSize[0];
                coords_out[1] = (coords_out[1] - (this.validArea[1]*canvasSize[1])) / validSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] /= validSize[0];
                    coords_out[3] /= validSize[1];
                }
            }

        } else {
            var canvasSize = [this.canvas[0].width, this.canvas[0].height];
            if(target === 'canvas') {
                coords_out[0] *= canvasSize[0];
                coords_out[1] *= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] *= canvasSize[0];
                    coords_out[3] *= canvasSize[1];
                }
            } else if(target === 'validArea') {
                var validSize = [this.validArea[2]*canvasSize[0], this.validArea[3]*canvasSize[1]];
                coords_out[0] = (coords_out[0]*validSize[0]) + (this.validArea[0]*canvasSize[0]);
                coords_out[1] = (coords_out[1]*validSize[1]) + (this.validArea[1]*canvasSize[1]);
                if(coords_out.length == 4) {
                    coords_out[2] *= validSize[0];
                    coords_out[3] *= validSize[1];
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

        // show loading text
        var loadingText = 'loading...';
        this.ctx.font = '20px sans-serif';
        var dimensions = this.ctx.measureText(loadingText);
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillText(loadingText, this.canvas[0].width/2 - dimensions.width/2, this.canvas[0].height/2);

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