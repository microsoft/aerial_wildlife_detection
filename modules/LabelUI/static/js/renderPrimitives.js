class AbstractRenderElement {

    constructor(id, zIndex) {
        this.id = id;
        this.zIndex = (zIndex == null? 0 : zIndex);
        this.isActive = false;
        this.changed = false;   // will be set to true if user modifies the initial geometry
    }

    setProperty(propertyName, value) {
        this[propertyName] = value;

        // set to user-modified
        this.changed = true;
    }

    getGeometry() {
        return {};
    }

    setActive(active, viewport) {
        this.isActive = active;
    }

    zIndex() {
        return this.zIndex;
    }

    render(ctx, viewport, limits, scaleFun) {
        throw Error('Not implemented.');
    }
}


class ElementGroup extends AbstractRenderElement {

    constructor(id, elements, zIndex) {
        super(id, zIndex);
        this.elements = elements;
        if(this.elements == null) {
            this.elements = [];
        }
    }

    addElement(element) {
        if(this.elements.indexOf(element) === -1) {
            this.elements.push(element);
        }
    }

    removeElement(element) {
        var idx = this.elements.indexOf(element);
        if(idx !== -1) {
            this.elements.splice(idx, 1);
        }
    }

    render(ctx, viewport, limits, scaleFun) {
        for(var e=0; e<this.elements.length; e++) {
            this.elements[e].render(ctx, viewport, limits, scaleFun);
        }
    }
}


class ImageElement extends AbstractRenderElement {

    constructor(id, viewport, imageURI, zIndex) {
        super(id, zIndex);
        this.viewport = viewport;
        this.imageURI = imageURI;
        // default parameter until image is loaded
        this.bounds = [0, 0, 1, 1];
        this._create_image();
    }

    _create_image() {
        this.image = new Image();
        // this.image.width = this.width;
        // this.image.height = this.height;
        var self = this;
        this.image.onload = function() {
            // calculate relative image offsets (X and Y)
            var canvasSize = [self.viewport.canvas.width(), self.viewport.canvas.height()];
            var scale = Math.min(canvasSize[0] / this.naturalWidth, canvasSize[1] / this.naturalHeight);
            var offset = [(canvasSize[0] - (scale*this.naturalWidth))/2, (canvasSize[1] - (scale*this.naturalHeight))/2];
            var sizeX = this.naturalWidth * scale;
            var sizeY = this.naturalHeight * scale;
            self.bounds = [offset[0]/canvasSize[0], offset[1]/canvasSize[1], sizeX/canvasSize[0], sizeY/canvasSize[1]];

            // define valid canvas area as per image offset
            self.viewport.setValidArea(self.bounds);

            // re-render
            self.viewport.render();
        };
        this.image.src = this.imageURI;
    }

    getNaturalImageExtent() {
        return this.naturalImageExtent;
    }

    render(ctx, viewport, limits, scaleFun) {
        var targetCoords = scaleFun(this.bounds, 'canvas');
        ctx.drawImage(this.image, targetCoords[0], targetCoords[1],
            targetCoords[2],
            targetCoords[3]);
    }
}



class HoverTextElement extends AbstractRenderElement {

    constructor(id, hoverText, position, zIndex) {
        super(id, zIndex);
        this.text = hoverText;
        this.position = position;
    }

    render(ctx, viewport, limits, scaleFun) {
        if(this.text == null) return;
        var hoverPos = scaleFun(this.position, 'canvas');
        var dimensions = ctx.measureText(this.text);
        dimensions.height = window.styles.hoverText.box.height;
        // dimensions = scaleFun([dimensions.width, dimensions.height]);
        var offsetH = window.styles.hoverText.offsetH;
        ctx.fillStyle = window.styles.hoverText.box.fill;
        ctx.fillRect(offsetH+hoverPos[0]-2, hoverPos[1]-(dimensions[1]/2+2), dimensions[0]+4, dimensions[1]+4);
        ctx.strokeStyle = window.styles.hoverText.box.stroke.color;
        ctx.lineWidth = window.styles.hoverText.box.stroke.lineWidth;
        ctx.strokeRect(offsetH+hoverPos[0]-2, hoverPos[1]-(dimensions[1]/2+2), dimensions[0]+4, dimensions[1]+4);
        ctx.fillStyle = window.styles.hoverText.text.color;
        ctx.font = window.styles.hoverText.text.font;
        ctx.fillText(this.text, offsetH+hoverPos[0], hoverPos[1]);
    }
}



class PointElement extends AbstractRenderElement {

    constructor(id, x, y, color, size, zIndex) {
        super(id, zIndex);
        this.x = x;
        this.y = y;
        this.color = color;
        this.size = size;
    }

    //TODO: add interactions

    getGeometry() {
        return {
            'type': 'point',
            'coordinates': [this.x, this.y]
        };
    }

    render(ctx, viewport, limits, scaleFun) {
        if(this.x == null || this.y == null) return;

        var coords = [this.x, this.y];

        // shift coordinates w.r.t. bounds
        coords[0] += limits[0];
        coords[1] += limits[1];

        coords = scaleFun(coords);

        var sz = scaleFun(this.size, this.size);
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(coords[0], coords[1], sz[0], 0, 2*Math.PI);
        ctx.fill();
        ctx.closePath();
    }

    euclideanDistance(that) {
        return Math.sqrt(Math.pow(this.x - that[0],2) + Math.pow(this.y - that[1],2));
    }
}



class LineElement extends AbstractRenderElement {

    constructor(id, startX, startY, endX, endY, strokeColor, lineWidth, lineDash, zIndex) {
        super(id, zIndex);
        this.startX = startX;
        this.startY = startY;
        this.endX = endX;
        this.endY = endY;
        this.strokeColor = strokeColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);
    }

    //TODO: add interactions

    getGeometry() {
        return {
            'type': 'line',
            'coordinates': [this.startX, this.startY, this.endX, this.endY]
        };
    }

    render(ctx, viewport, limits, scaleFun) {
        if(this.startX == null || this.startY == null ||
            this.endX == null || this.endY == null)
            return;
        
        var startPos = scaleFun([this.startX, this.startY], 'canvas');
        var endPos = scaleFun([this.endX, this.endY], 'canvas');
        
        if(this.strokeColor != null) ctx.strokeStyle = this.strokeColor;
        if(this.lineWidth != null) ctx.lineWidth = this.lineWidth;
        ctx.setLineDash(this.lineDash);
        ctx.beginPath();
        ctx.moveTo(startPos[0], startPos[1]);
        ctx.lineTo(endPos[0], endPos[1]);
        ctx.stroke();
        ctx.closePath();
    }
}


class RectangleElement extends PointElement {

    constructor(id, x, y, width, height, fillColor, strokeColor, lineWidth, lineDash, zIndex) {
        super(id, x, y, strokeColor, null, zIndex);
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.fillColor = fillColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);

        this.isActive = false;
    }

    getGeometry() {
        return {
            'type': 'rectangle',
            'coordinates': [this.x, this.y, this.width, this.height]
        };
    }

    registerAsCallback(viewport) {
        /*
            Adds this instance to the viewport.
            This makes the rectangle user-modifiable in terms of size and position.
        */
        viewport.addCallback(this.id, 'click', this._get_active_handle_callback('click', viewport));
    }

    deregisterAsCallback(viewport) {
        this.setActive(false, viewport);
        viewport.removeCallback(this.id, 'click');
    }

    getExtent() {
        return [this.x - this.width/2, this.y - this.height/2, this.x + this.width/2, this.y + this.height/2];
    }

    containsPoint(coordinates) {
        var extent = this.getExtent();
        return (coordinates[0] >= extent[0] && coordinates[0] <= extent[2]) &&
            (coordinates[1] >= extent[1] && coordinates[1] <= extent[3]);
    }

    isInDistance(coordinates, tolerance, forceCorner) {
        /*
            Returns true if any parts of the bounding box are
            within a tolerance's distance of the provided coordinates.
            If 'forceCorner' is true, coordinates have to be within
            reach of one of the bounding box's corners.
        */
        if(forceCorner) {
            return (this.getClosestHandle(coordinates, tolerance) != null);

        } else {
            var extentsTolerance = [this.x-this.width/2, this.y-this.height/2, this.x+this.width/2, this.y+this.height/2];
            return (coordinates[0] >= extentsTolerance[0] && coordinates[0] <= extentsTolerance[2]) &&
                (coordinates[1] >= extentsTolerance[1] && coordinates[1] <= extentsTolerance[3]);
        }
    }

    _createResizeHandles() {
        /*
            Returns small drawable rectangles at the corners
            and sides of the rectangle.
        */
        if(this.resizeHandles != null) {
            return this.resizeHandles;
        }

        var self = this;
        var getHandle = function(x, y) {
            return new ResizeHandle(
                self.id + '_resize_' + x + '_' + y,
                x, y,
                1);
        }
        var handles = [];

        // corners
        handles.push(getHandle(this.x - this.width/2, this.y - this.height/2));
        handles.push(getHandle(this.x - this.width/2, this.y + this.height/2));
        handles.push(getHandle(this.x + this.width/2, this.y - this.height/2));
        handles.push(getHandle(this.x + this.width/2, this.y + this.height/2));

        // sides
        handles.push(getHandle(this.x, this.y - this.height/2));
        handles.push(getHandle(this.x, this.y + this.height/2));
        handles.push(getHandle(this.x - this.width/2, this.y));
        handles.push(getHandle(this.x + this.width/2, this.y));
        
        this.resizeHandles = new ElementGroup(this.id + '_resizeHandles', handles);
    }

    _updateResizeHandles() {
        if(this.resizeHandles == null) return;
        this.resizeHandles.elements[0].setProperty('x', this.x - this.width/2);
        this.resizeHandles.elements[0].setProperty('y', this.y - this.height/2);
        this.resizeHandles.elements[1].setProperty('x', this.x - this.width/2);
        this.resizeHandles.elements[1].setProperty('y', this.y + this.height/2);
        this.resizeHandles.elements[2].setProperty('x', this.x + this.width/2);
        this.resizeHandles.elements[2].setProperty('y', this.y - this.height/2);
        this.resizeHandles.elements[3].setProperty('x', this.x + this.width/2);
        this.resizeHandles.elements[3].setProperty('y', this.y + this.height/2);
        this.resizeHandles.elements[4].setProperty('x', this.x);
        this.resizeHandles.elements[4].setProperty('y', this.y - this.height/2);
        this.resizeHandles.elements[5].setProperty('x', this.x);
        this.resizeHandles.elements[5].setProperty('y', this.y + this.height/2);
        this.resizeHandles.elements[6].setProperty('x', this.x - this.width/2);
        this.resizeHandles.elements[6].setProperty('y', this.y);
        this.resizeHandles.elements[7].setProperty('x', this.x + this.width/2);
        this.resizeHandles.elements[7].setProperty('y', this.y);
    }

    getClosestHandle(coordinates, tolerance) {
        /*
            Returns one of {'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'} if the coordinates
            are close to one of the adjustment handles within a given tolerance.
            Returns 'c' if coordinates are not close to handle, but within bounding box.
            Else returns null.
        */
        var matchL = Math.abs((this.x - this.width/2) - coordinates[0]) <= tolerance;
        var matchT = Math.abs((this.y - this.height/2) - coordinates[1]) <= tolerance;
        var matchR = Math.abs((this.x + this.width/2) - coordinates[0]) <= tolerance;
        var matchB = Math.abs((this.y + this.height/2) - coordinates[1]) <= tolerance;

        if(matchT) {
            if(matchL) return 'nw';
            if(matchR) return 'ne';
            return 'n';
        } else if(matchB) {
            if(matchL) return 'sw';
            if(matchR) return 'se';
            return 's';
        } else if(matchL) {
            return 'w';
        } else if(matchR) {
            return 'e';
        } else if(this.containsPoint(coordinates)) {
            return 'c';
        } else {
            return null;
        }
    }

    
    /* interaction events */
    _click_event(event, viewport) {
        if(window.interfaceControls.action != window.interfaceControls.actions.DO_NOTHING) return;
        var mousePos = viewport.getRelativeCoordinates(event, 'validArea');
        this.activeHandle = this.getClosestHandle(mousePos, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        if(this.activeHandle == null) {
            this.setActive(false, viewport);
        } else {
            if(!this.active) {
                this.setActive(true, viewport);
            }
        }
    }

    _mousedown_event(event, viewport) {
        this.mousePos_current = viewport.getRelativeCoordinates(event, 'validArea');
        this.mouseDrag = true;
        this.activeHandle = this.getClosestHandle(this.mousePos_current, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
    }

    _mousemove_event(event, viewport) {
        /*
            On mousemove, we update the target coordinates and the bounding box:
            - always: update cursor
            - if drag and close to resize handle: resize rectangle and move resize handles
            - if drag and inside rectangle: move rectangle and resize handles
        */
        var coords = viewport.getRelativeCoordinates(event, 'validArea');
        if(this.mousePos_current == null) {
            this.mousePos_current = coords;
        }
        var mpc = this.mousePos_current;
        var extent = this.getExtent();
        if(this.mouseDrag && this.activeHandle != null) {
            // move or resize rectangle
            if(this.activeHandle.includes('w')) {
                var width = extent[2] - mpc[0];
                if(width < 0) {
                    this.activeHandle = this.activeHandle.replace('w', 'e');
                }
                var x = mpc[0] + width/2;
                this.setProperty('width', width);
                this.setProperty('x', x);
            }
            if(this.activeHandle.includes('e')) {
                var width = mpc[0] - extent[0];
                if(width < 0) {
                    this.activeHandle = this.activeHandle.replace('e', 'w');
                }
                var x = mpc[0] - width/2;
                this.setProperty('width', width);
                this.setProperty('x', x);
            }
            if(this.activeHandle.includes('n')) {
                var height = extent[3] - mpc[1];
                if(height < 0) {
                    this.activeHandle = this.activeHandle.replace('n', 's');
                }
                var y = mpc[1] + height/2;
                this.setProperty('height', height);
                this.setProperty('y', y);
            }
            if(this.activeHandle.includes('s')) {
                var height = mpc[1] - extent[1];
                if(height < 0) {
                    this.activeHandle = this.activeHandle.replace('s', 'n');
                }
                var y = mpc[1] - height/2;
                this.setProperty('height', height);
                this.setProperty('y', y);
            }
            if(this.activeHandle.includes('c')) {
                this.setProperty('x', this.x + coords[0] - mpc[0]);
                this.setProperty('y', this.y + coords[1] - mpc[1]);
            }
        } else {
            this.activeHandle = this.getClosestHandle(mpc, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        }

        // update resize handles
        this._updateResizeHandles();

        // update cursor
        if(this.activeHandle == null) {
            viewport.canvas.css('cursor', 'crosshair');     //TODO: default cursor?
        } else if(this.activeHandle == 'c') {
            viewport.canvas.css('cursor', 'move');
        } else {
            viewport.canvas.css('cursor', this.activeHandle + '-resize');
        }

        // update current mouse pos
        this.mousePos_current = coords;

        // set to user-modified
        this.changed = true;
    }

    _mouseup_event(event, viewport) {
        this.mouseDrag = false;
    }


    _get_active_handle_callback(type, viewport) {
        var self = this;
        if(type == 'click') {
            /*
                Activates or deactivates this rectangle, depending on where
                the click landed.
            */
            return function(event) {
                self._click_event(event, viewport);
            }

        } else if(type == 'mousedown') {
            return function(event) {
                self._mousedown_event(event, viewport);
            };

        } else if(type == 'mousemove') {
            return function(event) {
                self._mousemove_event(event, viewport);
            }

        } else if(type == 'mouseup') {
            return function(event) {
                self._mouseup_event(event, viewport);
            }
        }
    }

    setActive(active, viewport) {
        /*
            Sets the 'active' property to the given value.
            Also draws resize handles to the viewport if active
            and makes them resizable through callbacks.
        */
        super.setActive(active, viewport);
        if(active) {
            this._createResizeHandles();
            viewport.addRenderElement(this.resizeHandles);
            // viewport.addCallback(this, 'click', this._get_active_handle_callback('click', viewport));
            viewport.addCallback(this.id, 'mousedown', this._get_active_handle_callback('mousedown', viewport));
            viewport.addCallback(this.id, 'mousemove', this._get_active_handle_callback('mousemove', viewport));
            viewport.addCallback(this.id, 'mouseup', this._get_active_handle_callback('mouseup', viewport));
        } else {
            viewport.removeRenderElement(this.resizeHandles);
            viewport.removeCallback(this.id, 'click');
            viewport.removeCallback(this.id, 'mousedown');
            viewport.removeCallback(this.id, 'mousemove');
            viewport.removeCallback(this.id, 'mouseup');
            this.mouseDrag = false;
        }
    }


    render(ctx, viewport, limits, scaleFun) {
        if(this.x == null || this.y == null) return;

        var coords = [this.x-this.width/2, this.y-this.height/2, this.width, this.height];
        coords = scaleFun(coords, 'validArea');
        if(this.fillColor != null) {
            ctx.fillStyle = this.fillColor;
            ctx.fillRect(coords[0], coords[1], coords[2], coords[3]);
        }
        if(this.color != null) {
            ctx.strokeStyle = this.color;
            ctx.lineWidth = this.lineWidth;
            ctx.setLineDash(this.lineDash);
            ctx.beginPath();
            ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
            ctx.closePath();
        }
    }
}



class BorderStrokeElement extends AbstractRenderElement {
    /*
        Draws a border around the viewport.
        Specifically intended for classification tasks.
    */
    constructor(id, strokeColor, lineWidth, lineDash, zIndex) {
        super(id, zIndex);
        this.color = strokeColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);
    }

    render(ctx, viewport, limits, scaleFun) {
        if(this.color == null) return;
        var coords = scaleFun(viewport, 'canvas');
        ctx.strokeStyle = this.color;
        ctx.lineWidth = this.lineWidth;
        ctx.setLineDash(this.lineDash);
        ctx.beginPath();
        ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
        ctx.closePath();
    }
}



class ResizeHandle extends AbstractRenderElement {
    /*
        Draws a small square at a given position that is fixed in size
        (but not in position), irrespective of scale.
    */
    constructor(id, x, y, zIndex) {
        super(id, zIndex);
        this.x = x;
        this.y = y;
    }

    render(ctx, viewport, limits, scaleFun) {
        if(this.x == null || this.y == null) return;

        var coords = [this.x, this.y];

        // adjust coordinates w.r.t. bounds
        // coords[0] = (coords[0] * limits[2]) + limits[0];
        // coords[1] = (coords[1] * limits[3]) + limits[1];

        coords = scaleFun(coords, 'validArea');

        var sz = window.styles.resizeHandles.size;

        ctx.fillStyle = window.styles.resizeHandles.fillColor;
        ctx.fillRect(coords[0] - sz/2, coords[1] - sz/2, sz, sz);
        ctx.strokeStyle = window.styles.resizeHandles.strokeColor;
        ctx.lineWidth = window.styles.resizeHandles.lineWidth;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.strokeRect(coords[0] - sz/2, coords[1] - sz/2, sz, sz);
        ctx.closePath();
    }
}