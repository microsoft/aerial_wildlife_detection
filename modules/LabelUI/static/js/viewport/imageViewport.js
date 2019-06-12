/*
    Interface to the HTML canvas for regular image annotations.

    2019 Benjamin Kellenberger
*/

class ImageViewport {

    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas[0].getContext('2d');
        this.viewport = [0, 0, canvas[0].width, canvas[0].height];
        this.renderStack = [];
        this.renderStack.sort(function(a, b) {
            if(a.zIndex() < b.zIndex()) {
                return -1;
            } else if(a.zIndex() > b.zIndex()) {
                return 1;
            } else {
                return 0;
            }
        });
    }

    _getCanvasScaleFactors() {
        var scaleX = window.defaultImage_w / this.viewport[2];
        var scaleY = window.defaultImage_h / this.viewport[3];
        return [scaleX, scaleY];
    }

    addRenderElement(element) {
        if(this.indexOfRenderElement(element) === -1) {
            this.renderStack.push(element);
            this.renderStack.sort();
            this.render();
        }
    }

    indexOfRenderElement(element) {
        return(this.renderStack.indexOf(element));
    }

    updateRenderElement(index, element) {
        this.renderStack[index] = element;
        this.renderStack.sort();
        this.render();
    }

    removeRenderElement(element) {
        var idx = this.renderStack.indexOf(element);
        if(idx !== -1) {
            this.renderStack.splice(idx, 1);
            this.render();
        }
    }

    getCanvasCoordinates(event, scaled) {
        var posX = (event.pageX - this.canvas.offset().left);
        var posY = (event.pageY - this.canvas.offset().top);
        var coords = [posX, posY];
        if(scaled) {
            coords = this.scaleToCanvas(coords);
        }
        return coords;
    }

    scaleToCanvas(coordinates) {
        /*
            Accepts given coordinates ([X, Y] or [X, Y, W, H])
            and returns versions that are scaled and shifted to
            match the current viewport.
        */
        var scaleFactors = this._getCanvasScaleFactors();
        var coordsOut = [];
        coordsOut.push((coordinates[0] - this.viewport[0]) * scaleFactors[0]);     //TODO: check
        coordsOut.push((coordinates[1] - this.viewport[1]) * scaleFactors[1]);     //TODO: check

        if(coordinates.length == 4) {
            coordsOut.push(coordinates[2] * scaleFactors[0]);
            coordsOut.push(coordinates[3] * scaleFactors[1]);
        }
        return coordsOut;
    }

    render() {
        // clear canvas
        var extent = this.scaleToCanvas([0, 0, this.canvas[0].width, this.canvas[0].height]);
        this.ctx.fillStyle = window.styles.background;
        this.ctx.fillRect(0, 0, extent[2], extent[3]);

        // iterate through render stack
        var self = this;
        var scaleFun = function(coords) {
            return self.scaleToCanvas(coords);
        }
        for(var i=0; i<this.renderStack.length; i++) {
            this.renderStack[i].render(this.ctx, this.viewport, scaleFun);
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
}