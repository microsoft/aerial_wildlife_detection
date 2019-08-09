function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    /**
     * Draws a rounded rectangle using the current state of the canvas.
     * If you omit the last three params, it will draw a rectangle
     * outline with a 5 pixel border radius
     * @param {CanvasRenderingContext2D} ctx
     * @param {Number} x The top left x coordinate
     * @param {Number} y The top left y coordinate
     * @param {Number} width The width of the rectangle
     * @param {Number} height The height of the rectangle
     * @param {Number} [radius = 5] The corner radius; It can also be an object 
     *                 to specify different radii for corners
     * @param {Number} [radius.tl = 0] Top left
     * @param {Number} [radius.tr = 0] Top right
     * @param {Number} [radius.br = 0] Bottom right
     * @param {Number} [radius.bl = 0] Bottom left
     * @param {Boolean} [fill = false] Whether to fill the rectangle.
     * @param {Boolean} [stroke = true] Whether to stroke the rectangle.
     * 
     * source: https://stackoverflow.com/questions/1255512/how-to-draw-a-rounded-rectangle-on-html-canvas
     */
    if (typeof stroke == 'undefined') {
      stroke = true;
    }
    if (typeof radius === 'undefined') {
      radius = 5;
    }
    if (typeof radius === 'number') {
      radius = {tl: radius, tr: radius, br: radius, bl: radius};
    } else {
      var defaultRadius = {tl: 0, tr: 0, br: 0, bl: 0};
      for (var side in defaultRadius) {
        radius[side] = radius[side] || defaultRadius[side];
      }
    }
    ctx.beginPath();
    ctx.moveTo(x + radius.tl, y);
    ctx.lineTo(x + width - radius.tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
    ctx.lineTo(x + width, y + height - radius.br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
    ctx.lineTo(x + radius.bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
    ctx.lineTo(x, y + radius.tl);
    ctx.quadraticCurveTo(x, y, x + radius.tl, y);
    ctx.closePath();
    if (fill) {
      ctx.fill();
    }
    if (stroke) {
      ctx.stroke();
    }
  
  }



class AbstractRenderElement {

    constructor(id, style, zIndex, disableInteractions) {
        this.id = id;
        this.style = style;
        if(this.style === null || this.style === undefined) {
            this.style = {};
        }
        this.style.setProperty = function(propertyName, value) {
            if(this.hasOwnProperty(propertyName)) {
                this[propertyName] = value;
            }
        }
        this.zIndex = (zIndex == null? 0 : zIndex);
        this.disableInteractions = disableInteractions;
        this.isActive = false;
        this.changed = false;   // will be set to true if user modifies the initial geometry
        this.lastUpdated = new Date();  // timestamp of last update
        this.isValid = true;    // set to false by instances that are missing required properties (e.g. coordinates)
        this.unsure = false;
        this.visible = true;
    }

    getProperty(propertyName) {
        if(this.hasOwnProperty(propertyName)) {
            return this[propertyName];
        } else if(this.style.hasOwnProperty(propertyName)) {
            return this.style[propertyName];
        }
        return null;
    }

    setProperty(propertyName, value) {

        // handle style properties separately
        this.style.setProperty(propertyName, value);

        this[propertyName] = value;

        // set to user-modified
        if(!['id', 'isActive', 'visible', 'zIndex'].includes(propertyName)) {
            this.changed = true;
            this.lastUpdated = new Date();
        }
    }

    getGeometry() {
        return {
            'unsure': this.unsure
        };
    }

    getLastUpdated() {
        return this.lastUpdated;
    }

    setActive(active, viewport) {
        this.isActive = active;
    }

    zIndex() {
        return this.zIndex;
    }

    setVisible(visible) {
        this.visible = visible;
    }

    render(ctx, scaleFun) {
        if(!this.visible) return;
    }
}


class ElementGroup extends AbstractRenderElement {

    constructor(id, elements, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
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

    setVisible(visible) {
        super.setVisible(visible);

        for(var e=0; e<this.elements.length; e++) {
            this.elements[e].setVisible(visible);
        }
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        for(var e=0; e<this.elements.length; e++) {
            this.elements[e].render(ctx, scaleFun);
        }
    }
}


class ImageElement extends AbstractRenderElement {

    constructor(id, viewport, imageURI, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
        this.viewport = viewport;
        this.imageURI = imageURI;
        // default parameter until image is loaded
        this.bounds = [0, 0, 1, 1];
        this._create_image();
    }

    _create_image() {
        this.image = new Image();
        this.image.loadingText = 'loading image...';
        var self = this;
        this.image.onload = function() {
            // calculate image bounds
            var limit = Math.max(this.naturalWidth, this.naturalHeight);
            
            var width = this.naturalWidth / limit;
            var height = this.naturalHeight / limit;

            self.bounds = [(1-width)/2, (1-height)/2, width, height];

            // define valid canvas area as per image offset
            self.viewport.setValidArea(self.bounds);

            // set time created
            self.timeCreated = new Date();
            this.loaded = true;

            // re-render
            self.viewport.render();
        };
        this.image.onerror = function(e) {
            this.loaded = false;
            self.image.loadingText = 'loading failed.';
        };
        this.image.src = this.imageURI;
    }

    getNaturalImageExtent() {
        return this.naturalImageExtent;
    }

    getTimeCreated() {
        return this.timeCreated;
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        var targetCoords = scaleFun([0,0,1,1], 'validArea');
        if(this.image.loaded) {
            ctx.drawImage(this.image, targetCoords[0], targetCoords[1],
                targetCoords[2],
                targetCoords[3]);

        } else {
            // loading failed
            ctx.fillStyle = window.styles.background;
            ctx.fillRect(targetCoords[0], targetCoords[1], targetCoords[2], targetCoords[3]);
            ctx.font = '20px sans-serif';
            var dimensions = ctx.measureText(this.image.loadingText);
            ctx.fillStyle = '#FFFFFF';
            ctx.fillText(this.image.loadingText, targetCoords[2]/2 - dimensions.width/2, targetCoords[3]/2);
        }
    }
}



class HoverTextElement extends AbstractRenderElement {

    constructor(id, hoverText, position, reference, style, zIndex, disableInteractions) {
        super(id, style, zIndex, disableInteractions);
        this.text = hoverText;
        this.position = position;
        this.reference = reference;
        if(this.style.textColor == null || this.style.textColor == undefined) {
            this.style['textColor'] = window.styles.hoverText.text.color;
        }
    }

    setProperty(propertyName, value) {
        super.setProperty(propertyName, value);
        if(propertyName === 'color') {
            this.style.fillColor = window.addAlpha(value, this.style.fillOpacity);
        }
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        if(this.text == null) return;
        var hoverPos = scaleFun(this.position, this.reference);
        ctx.font = window.styles.hoverText.text.fontSizePix + 'px ' + window.styles.hoverText.text.fontStyle;
        var dimensions = ctx.measureText(this.text);
        dimensions.height = window.styles.hoverText.box.height;
        dimensions = [dimensions.width + 8, dimensions.height];
        var offsetH = window.styles.hoverText.offsetH;
        
        if(this.style.fillColor != null) {
            ctx.fillStyle = this.style.fillColor;
            ctx.fillRect(offsetH+hoverPos[0]-4, hoverPos[1]-(dimensions[1]/2+4), dimensions[0]+4, dimensions[1]+4);
        }
        if(this.style.strokeColor != null && this.style.lineWidth != null) {
            ctx.strokeStyle = this.style.strokeColor;
            ctx.lineWidth = this.style.lineWidth;
            ctx.setLineDash([]);
            ctx.strokeRect(offsetH+hoverPos[0]-4, hoverPos[1]-(dimensions[1]/2+4), dimensions[0]+4, dimensions[1]+4);
        }
        ctx.fillStyle = this.style.textColor;
        ctx.fillText(this.text, offsetH+hoverPos[0], hoverPos[1]);
    }
}



class PointElement extends AbstractRenderElement {

    constructor(id, x, y, style, unsure, zIndex, disableInteractions) {
        super(id, style, zIndex, disableInteractions);
        if(!this.style.hasOwnProperty('fillColor') && this.style.hasOwnProperty('color')) {
            this.style['fillColor'] = window.addAlpha(this.style.color, this.style.fillOpacity);
        }
        this.x = x;
        this.y = y;
        this.unsure = unsure;

        this.isValid = (x != null && y != null);
    }

    setProperty(propertyName, value) {
        super.setProperty(propertyName, value);
        if(propertyName === 'color') {
            this.style.fillColor = window.addAlpha(value, this.style.fillOpacity);
        }
    }

    getGeometry() {
        return {
            'type': 'point',
            'x': this.x,
            'y': this.y,
            'unsure': this.unsure
        };
    }

    /* interaction events */
    _mousedown_event(event, viewport, force) {
        if(!this.visible ||
            !force && (!([ACTIONS.DO_NOTHING, ACTIONS.ADD_ANNOTATION].includes(window.uiControlHandler.getAction())))) return;
        this.mousePos_current = viewport.getRelativeCoordinates(event, 'validArea');
        this.mouseDrag = (event.which === 1);
        var tolerance = viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
        if(this.euclideanDistance(this.mousePos_current) <= tolerance) {
            viewport.canvas.css('cursor', 'move');
        }
    }

    _mousemove_event(event, viewport, force) {
        /*
            On mousemove, we update the target coordinates and the point:
            - always: update cursor
            - if drag and within distance to point: move it
        */
        if(!this.visible) return;
        var coords = viewport.getRelativeCoordinates(event, 'validArea');
        if(this.mousePos_current == null) {
            this.mousePos_current = coords;
        }
        var mpc = this.mousePos_current;
        var tolerance = viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
        if(this.mouseDrag && this.euclideanDistance(mpc) <= tolerance) {
            // move point
            this.setProperty('x', this.x + coords[0] - mpc[0]);
            this.setProperty('y', this.y + coords[1] - mpc[1]);

            // update timestamp
            this.lastUpdated = new Date();

            // update cursor
            viewport.canvas.css('cursor', 'move');
        }

        // update current mouse pos
        this.mousePos_current = coords;

        // set to user-modified
        this.changed = true;
    }

    _mouseup_event(event, viewport, force) {
        this.mouseDrag = false;
        if(!this.visible ||
            !force && (!(window.uiControlHandler.getAction() === ACTIONS.DO_NOTHING ||
            window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION))) return;
        
        var mousePos = viewport.getRelativeCoordinates(event, 'validArea');
        
        // activate if position within tolerance
        var tolerance = viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
        if(this.euclideanDistance(mousePos) <= tolerance) {
            this.setActive(true, viewport);
        } else {
            this.setActive(false, viewport);
        }
    }

    _mouseleave_event(event, viewport, force) {
        this.mouseDrag = false;
        if(force || window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION) {
            window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);
        }
    }

    _get_active_handle_callback(type, viewport) {
        var self = this;
        if(type === 'mousedown') {
            return function(event) {
                self._mousedown_event(event, viewport, false);
            };

        } else if(type === 'mousemove') {
            return function(event) {
                self._mousemove_event(event, viewport, false);
            }

        } else if(type === 'mouseup') {
            return function(event) {
                self._mouseup_event(event, viewport, false);
            }
        } else if(type === 'mouseleave') {
            return function(event) {
                self._mouseleave_event(event, viewport, false);
            }
        }
    }

    setActive(active, viewport) {
        /*
            Sets the 'active' property to the given value.
        */
        if(!this.disableInteractions) {
            super.setActive(active, viewport);
            if(active) {
                viewport.addCallback(this.id, 'mousedown', this._get_active_handle_callback('mousedown', viewport));
                viewport.addCallback(this.id, 'mousemove', this._get_active_handle_callback('mousemove', viewport));
                viewport.addCallback(this.id, 'mouseup', this._get_active_handle_callback('mouseup', viewport));
                viewport.addCallback(this.id, 'mouseleave', this._get_active_handle_callback('mouseleave', viewport));
            } else {

                // remove active properties
                viewport.removeCallback(this.id, 'mousedown');
                viewport.removeCallback(this.id, 'mousemove');
                viewport.removeCallback(this.id, 'mouseup');
                viewport.removeCallback(this.id, 'mouseleave');
                this.mouseDrag = false;
            }
        }
    }

    registerAsCallback(viewport) {
        /*
            Adds this instance to the viewport.
            This makes the entry user-modifiable in terms of position.
        */
        if(!this.disableInteractions)
            viewport.addCallback(this.id, 'mouseup', this._get_active_handle_callback('mouseup', viewport));
    }

    deregisterAsCallback(viewport) {
        this.setActive(false, viewport);
        viewport.removeCallback(this.id, 'mouseup');
    }

    render(ctx, scaleFun) {
        if(!this.visible || this.x == null || this.y == null) return;

        var coords = scaleFun([this.x, this.y], 'validArea');

        // draw actual point
        ctx.fillStyle = window.addAlpha(this.style.fillColor, this.style.fillOpacity);
        ctx.beginPath();
        ctx.arc(coords[0], coords[1], this.style.pointSize, 0, 2*Math.PI);
        ctx.fill();
        ctx.closePath();

        // if active: also draw outline
        if(this.isActive) {
            ctx.strokeStyle = window.addAlpha(this.style.fillColor, this.style.lineOpacity);
            ctx.lineWidth = 4;
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.arc(coords[0], coords[1], this.style.pointSize + 6, 0, 2*Math.PI);
            ctx.stroke();
            ctx.closePath();
        }

        // unsure flag
        if(this.unsure) {
            var text = 'unsure';
            var scaleFactors = scaleFun([0,0,ctx.canvas.width,ctx.canvas.height], 'canvas', true).slice(2,4);
            ctx.font = window.styles.hoverText.text.fontSizePix * scaleFactors[0] + 'px ' + window.styles.hoverText.text.fontStyle;
            var dimensions = ctx.measureText(text);
            dimensions.height = window.styles.hoverText.box.height;
            dimensions = [dimensions.width + 8, dimensions.height * scaleFactors[1]];
            ctx.setLineDash([]);
            ctx.fillStyle = this.style.fillColor;
            ctx.fillRect(coords[0]+4, coords[1]-(dimensions[1]), dimensions[0]+8, dimensions[1]);
            ctx.fillStyle = window.styles.hoverText.text.color;
            ctx.fillText(text, coords[0]+12, coords[1]-dimensions[1]/2+4);
        }
    }

    euclideanDistance(that) {
        return Math.sqrt(Math.pow(this.x - that[0],2) + Math.pow(this.y - that[1],2));
    }

    isInDistance(coordinates, tolerance) {
        /*
            Returns true if the point is within a tolerance's distance
            of the provided coordinates.
        */
        return this.euclideanDistance(coordinates) <= tolerance;
    }
}



class LineElement extends AbstractRenderElement {

    constructor(id, startX, startY, endX, endY, style, unsure, zIndex, disableInteractions) {
        super(id, style, zIndex, disableInteractions);
        if(!this.style.hasOwnProperty('strokeColor') && this.style.hasOwnProperty('color')) {
            this.style['strokeColor'] = window.addAlpha(this.style.color, this.style.lineOpacity);
        }
        this.startX = startX;
        this.startY = startY;
        this.endX = endX;
        this.endY = endY;
        this.unsure = unsure;

        this.isValid = (startX != null && startY != null && endX != null && endY != null);
    }

    setProperty(propertyName, value) {
        super.setProperty(propertyName, value);
        if(propertyName === 'color') {
            this.style.strokeColor = window.addAlpha(value, this.style.lineOpacity);
        }
    }

    getGeometry() {
        return {
            'type': 'line',
            'startX': this.startX,
            'startY': this.startY,
            'endX': this.endX,
            'endY': this.endY,
            'unsure': this.unsure
        };
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        if(this.startX == null || this.startY == null ||
            this.endX == null || this.endY == null)
            return;
        
        var startPos = scaleFun([this.startX, this.startY], 'validArea');
        var endPos = scaleFun([this.endX, this.endY], 'validArea');
        
        if(this.style.strokeColor != null) ctx.strokeStyle = this.style.strokeColor;
        if(this.style.lineWidth != null) ctx.lineWidth = this.style.lineWidth;
        ctx.setLineDash(this.style.lineDash);
        ctx.beginPath();
        ctx.moveTo(startPos[0], startPos[1]);
        ctx.lineTo(endPos[0], endPos[1]);
        ctx.stroke();
        ctx.closePath();
    }
}


class RectangleElement extends PointElement {

    constructor(id, x, y, width, height, style, unsure, zIndex, disableInteractions) {
        super(id, x, y, style, unsure, zIndex, disableInteractions);
        if(!this.style.hasOwnProperty('strokeColor') && this.style.hasOwnProperty('color')) {
            this.style['strokeColor'] = window.addAlpha(this.style.color, this.style.lineOpacity);
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;

        this.isValid = (x != null && y != null && width != null && height != null);
        this.isActive = false;
    }

    setProperty(propertyName, value) {
        super.setProperty(propertyName, value);
        if(propertyName === 'color') {
            this.style.strokeColor = window.addAlpha(value, this.style.lineOpacity);
        }
    }

    getGeometry() {
        return {
            'type': 'rectangle',
            'x': this.x,
            'y': this.y,
            'width': this.width,
            'height': this.height,
            'unsure': this.unsure
        };
    }

    getExtent() {
        return [this.x - this.width/2, this.y - this.height/2, this.x + this.width/2, this.y + this.height/2];
    }

    containsPoint(coordinates) {
        var extent = this.getExtent();
        return (coordinates[0] >= extent[0] && coordinates[0] <= extent[2]) &&
            (coordinates[1] >= extent[1] && coordinates[1] <= extent[3]);
    }

    isInDistance(coordinates, tolerance) {
        /*
            Returns true if any parts of the bounding box are
            within a tolerance's distance of the provided coordinates.
        */
        var extentsTolerance = [this.x-this.width/2-tolerance, this.y-this.height/2-tolerance, this.x+this.width/2+tolerance, this.y+this.height/2+tolerance];
        return (coordinates[0] >= extentsTolerance[0] && coordinates[0] <= extentsTolerance[2]) &&
            (coordinates[1] >= extentsTolerance[1] && coordinates[1] <= extentsTolerance[3]);
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

        // check first if cursor is within reach
        if(!this.isInDistance(coordinates, tolerance)) return null;

        var distL = Math.abs((this.x - this.width/2) - coordinates[0]);
        var distT = Math.abs((this.y - this.height/2) - coordinates[1]);
        var distR = Math.abs((this.x + this.width/2) - coordinates[0]);
        var distB = Math.abs((this.y + this.height/2) - coordinates[1]);
        
        var distCorner = Math.min(distL, distT, distR, distB);
        var distCenter = Math.sqrt(Math.pow(this.x-coordinates[0], 2) + Math.pow(this.y-coordinates[1], 2));

        if(distCenter < distCorner && distCenter <= tolerance) {
            return 'c';

        } else if(distT <= tolerance) {
            if(distL <= tolerance) return 'nw';
            if(distR <= tolerance) return 'ne';
            return 'n';
        } else if(distB <= tolerance) {
            if(distL <= tolerance) return 'sw';
            if(distR <= tolerance) return 'se';
            return 's';
        } else if(distL <= tolerance) {
            return 'w';
        } else if(distR <= tolerance) {
            return 'e';
        } else if(this.containsPoint(coordinates)) {
            return 'c';
        } else {
            return null;
        }
    }

    
    /* interaction events */
    _mousedown_event(event, viewport, force) {
        if(!this.visible ||
            !force && (!([ACTIONS.DO_NOTHING, ACTIONS.ADD_ANNOTATION].includes(window.uiControlHandler.getAction())))) return;
        this.mousePos_current = viewport.getRelativeCoordinates(event, 'validArea');
        this.mouseDrag = (event.which === 1);
        this.activeHandle = this.getClosestHandle(this.mousePos_current, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        if(this.activeHandle === 'c') {
            // center of a box clicked; set globally so that other active boxes don't falsely resize
            viewport.canvas.css('cursor', 'move');
        }
    }

    _mousemove_event(event, viewport, force) {
        /*
            On mousemove, we update the target coordinates and the bounding box:
            - always: update cursor
            - if drag and close to resize handle: resize rectangle and move resize handles
            - if drag and inside rectangle: move rectangle and resize handles
        */
        if(!this.visible || 
            !force && (!(window.uiControlHandler.getAction() === ACTIONS.DO_NOTHING || window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION)
            )) return;
        var coords = viewport.getRelativeCoordinates(event, 'validArea');
        // var handle = this.getClosestHandle(coords, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        if(this.mousePos_current == null) {
            this.mousePos_current = coords;
        }
        var mpc = this.mousePos_current;
        var extent = this.getExtent();
        // if(this.activeHandle == null && handle == null && this.mouseDrag) {
        //     // clicked somewhere in a center of a box; move instead of resize
        //      //TODO: this allows moving rectangles even if mouse dragged out in the blue...
        //     this.setProperty('x', this.x + coords[0] - mpc[0]);
        //     this.setProperty('y', this.y + coords[1] - mpc[1]);

        //     // update timestamp
        //     this.lastUpdated = new Date();

        // } else
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

            // update timestamp
            this.lastUpdated = new Date();
        } else {
            this.activeHandle = this.getClosestHandle(mpc, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        }

        // update resize handles
        this._updateResizeHandles();

        // update cursor
        if(window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION || this.activeHandle == null) {
            viewport.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());
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

    _mouseup_event(event, viewport, force) {
        this._clamp_min_box_size(viewport);
        if(!this.visible ||
            !force && (!(window.uiControlHandler.getAction() === ACTIONS.DO_NOTHING ||
            window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION))) return;
        var mousePos = viewport.getRelativeCoordinates(event, 'validArea');
        this.activeHandle = this.getClosestHandle(mousePos, window.annotationProximityTolerance / Math.min(viewport.canvas.width(), viewport.canvas.height()));
        if(this.activeHandle == null) {
            this.setActive(false, viewport);
        } else {
            if(!this.active) {
                this.setActive(true, viewport);
            }
        }

        this.mouseDrag = false;
    }


    _clamp_min_box_size(viewport) {
        // Make sure box is of correct size
        var minWidth = window.minBoxSize_w;
        var minHeight = window.minBoxSize_h;
        var minSize = viewport.transformCoordinates([0, 0, minWidth, minHeight], 'validArea', true).slice(2,4);
        this.width = Math.max(this.width, minSize[0]);
        this.height = Math.max(this.height, minSize[1]);
    }


    setActive(active, viewport) {
        /*
            Sets the 'active' property to the given value.
            Also draws resize handles to the viewport if active
            and makes them resizable through callbacks.
        */
        if(this.disableInteractions) return;

        super.setActive(active, viewport);
        if(active) {
            this._createResizeHandles();
            viewport.addRenderElement(this.resizeHandles);
            viewport.addCallback(this.id, 'mousedown', this._get_active_handle_callback('mousedown', viewport));
            viewport.addCallback(this.id, 'mousemove', this._get_active_handle_callback('mousemove', viewport));
            viewport.addCallback(this.id, 'mouseup', this._get_active_handle_callback('mouseup', viewport));
            viewport.addCallback(this.id, 'mouseleave', this._get_active_handle_callback('mouseleave', viewport));
        } else {
            // catch and assert min. box size before disabling callback
            this._clamp_min_box_size(viewport);

            // remove active properties
            viewport.removeRenderElement(this.resizeHandles);
            viewport.removeCallback(this.id, 'mousedown');
            viewport.removeCallback(this.id, 'mousemove');
            viewport.removeCallback(this.id, 'mouseup');
            viewport.removeCallback(this.id, 'mouseleave');
            this.mouseDrag = false;
        }
    }


    setVisible(visible) {
        super.setVisible(visible);

        // also propagate to resize handles (if available)
        if(this.resizeHandles != null) {
            this.resizeHandles.setVisible(visible);
        }
    }


    render(ctx, scaleFun) {
        if(!this.visible || this.x == null || this.y == null) return;

        var coords = [this.x-this.width/2, this.y-this.height/2, this.width, this.height];
        coords = scaleFun(coords, 'validArea');
        if(this.style.fillColor != null) {
            ctx.fillStyle = this.style.fillColor;
            ctx.fillRect(coords[0], coords[1], coords[2], coords[3]);
        }
        if(this.style.strokeColor != null) {
            ctx.strokeStyle = this.style.strokeColor;
            ctx.lineWidth = this.style.lineWidth;
            ctx.setLineDash(this.style.lineDash);
            ctx.beginPath();
            ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
            ctx.closePath();
        }
        if(this.unsure) {
            var text = 'unsure';
            var scaleFactors = scaleFun([0,0,ctx.canvas.width,ctx.canvas.height], 'canvas', true).slice(2,4);
            ctx.font = window.styles.hoverText.text.fontSizePix * scaleFactors[0] + 'px ' + window.styles.hoverText.text.fontStyle;
            var dimensions = ctx.measureText(text);
            dimensions.height = window.styles.hoverText.box.height;
            dimensions = [dimensions.width + 8, dimensions.height * scaleFactors[1]];
            ctx.setLineDash([]);
            ctx.fillStyle = this.style.strokeColor;
            ctx.fillRect(coords[0]-4, coords[1]-(dimensions[1]), dimensions[0]+4, dimensions[1]);
            ctx.fillStyle = window.styles.hoverText.text.color;
            ctx.fillText(text, coords[0]+4, coords[1]-dimensions[1]/2+4);
        }
    }
}



class BorderStrokeElement extends AbstractRenderElement {
    /*
        Draws a border around the viewport.
        Specifically intended for classification tasks.
    */
    constructor(id, text, style, unsure, zIndex, disableInteractions) {
        super(id, style, zIndex, disableInteractions);
        if(this.style.textColor == null || this.style.textColor == undefined) {
            this.style['textColor'] = window.styles.hoverText.text.color;
        }
        this.text = text;
        this.unsure = unsure;
        this.changed = true;        // always true; we want to collect all classification entries, since user will screen them anyway
    }

    setProperty(propertyName, value) {
        super.setProperty(propertyName, value);
        if(propertyName === 'color') {
            this.style.strokeColor = window.addAlpha(value, this.style.lineOpacity);
            this.style.fillColor = window.addAlpha(value, this.style.fillOpacity);
            if(value == null || value == undefined) {
                this.text = null;
            }
        }
    }

    getGeometry() {
        return {
            'type': 'label',
            'unsure': this.unsure
        };
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        if(!this.visible) return;
        var scaleFactors = scaleFun([ctx.canvas.width,ctx.canvas.height], 'canvas', true);
        var coords = scaleFun([0,0,1,1], 'canvas');

        if(this.style.strokeColor != null) {
            ctx.strokeStyle = this.style.strokeColor;
            ctx.lineWidth = this.style.lineWidth * scaleFactors[0];
            ctx.setLineDash(this.style.lineDash);
            ctx.beginPath();
            ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
            ctx.closePath();
        }

        // show text in bottom left corner
        var text = ''
        if(this.text != null) text = this.text + ' ';
        if(this.unsure) text += '(unsure)';
        if(this.text != null || this.unsure) {
            text = text.trim();
            ctx.fillStyle = window.styles.hoverText.text.color;
            ctx.font = window.styles.hoverText.text.fontSizePix * scaleFactors[0] + 'px ' + window.styles.hoverText.text.fontStyle;
            var dimensions = ctx.measureText(text);
            dimensions.height = window.styles.hoverText.box.height;
            dimensions = [dimensions.width + 8, dimensions.height * scaleFactors[1]]
            ctx.fillStyle = (this.style.strokeColor == null ? '#929292' : this.style.strokeColor);
            ctx.fillRect(coords[0], coords[3] - dimensions[1]/2 - 4, dimensions[0], dimensions[1] + 8);
            ctx.fillStyle = this.style.textColor;
            ctx.fillText(text, coords[0] + 4, coords[3] - 4);
        }
    }
}



class ResizeHandle extends AbstractRenderElement {
    /*
        Draws a small square at a given position that is fixed in size
        (but not in position), irrespective of scale.
    */
    constructor(id, x, y, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
        this.x = x;
        this.y = y;
    }

    render(ctx, scaleFun) {
        super.render(ctx, scaleFun);
        if(!this.visible || this.x == null || this.y == null) return;

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




class MiniViewport extends AbstractRenderElement {
    /*
        Miniature version of the viewport to be displayed at a given size
        and position on the parent viewport.
        Useful when only a sub-part of the viewport's area is to be shown.
    */
    constructor(id, parentViewport, parentExtent, x, y, size, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
        this.parentViewport = parentViewport;
        this.parentExtent = parentExtent;
        this.position = null;
        if(x != null && y != null && size != null)
            this.position = [x, y, size, size];
    }

    getPosition() {
        return this.position;
    }

    setPosition(x, y, size) {
        this.position = [x, y, size, size];
    }


    getParentExtent() {
        return this.parentExtent;
    }

    setParentExtent(extent) {
        this.parentExtent = extent;
    }


    render(ctx, scaleFun) {
        if(this.position == null || this.parentExtent == null) return;
        super.render(ctx, scaleFun);

        // draw parent canvas as an image
        var pos_abs = scaleFun(this.position, 'canvas');
        var parentExtent_abs = scaleFun(this.parentExtent, 'canvas');
        ctx.drawImage(
            this.parentViewport.canvas[0],
            parentExtent_abs[0], parentExtent_abs[1],
            parentExtent_abs[2]-parentExtent_abs[0], parentExtent_abs[3]-parentExtent_abs[1],
            pos_abs[0], pos_abs[1],
            pos_abs[2], pos_abs[3]
        )
    }
}




class MiniMap extends AbstractRenderElement {
    /*
        The MiniMap, unlike the MiniViewport, actually re-draws the
        canvas elements and also offers an interactive rectangle, showing
        the position of the parent viewport's current extent.
    */
    constructor(id, parentViewport, x, y, size, interactive, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
        this.parentViewport = parentViewport;

        this.position = null;
        if(x != null && y != null && size != null)
            this.position = [x, y, size, size];

        if(interactive)
            this._setup_interactions();
    }

    _mousedown_event(event) {
        if(this.position == null || this.pos_abs == null) return;

        // check if mousedown over mini-rectangle
        var mousePos = this.parentViewport.getAbsoluteCoordinates(event);
        var extent_parent = this.minimapScaleFun(this.parentViewport.getViewport(), 'canvas');
        
        //TODO: something's still buggy here...

        if(mousePos[0] >= extent_parent[0] && mousePos[1] >= extent_parent[1] &&
            mousePos[0] < (extent_parent[0]+extent_parent[1]) &&
            mousePos[1] < (extent_parent[1]+extent_parent[2])) {

            this.mousePos = mousePos;
            this.mouseDown = true;
        }
    }

    _mousemove_event(event) {
        if(this.position == null || this.pos_abs == null || !this.mouseDown) return;

        // determine difference to previous parent extent
        var newMousePos = this.parentViewport.getAbsoluteCoordinates(event);
        var diffX = (newMousePos[0] - this.mousePos[0]);
        var diffY = (newMousePos[1] - this.mousePos[1]);
        this.mousePos = newMousePos;

        // backproject differences to full canvas size
        var diffProj = this.minimapScaleFun([0, 0, diffX, diffY], 'canvas', true).slice(2,4);
        var vp = this.parentViewport.getViewport();

        // apply new viewport
        vp[0] += diffProj[0];
        vp[1] += diffProj[1];
        this.parentViewport.setViewport(vp);
    }
    
    _mouseup_event(event) {
        this.mouseDown = false;
    }

    _mouseleave_event(event) {
        this.mouseDown = false;
    }

    __get_callback(type) {
        var self = this;
        if(type === 'mousedown') {
            return function(event) {
                self._mousedown_event(event);
            };
        } else if(type ==='mousemove') {
            return function(event) {
                self._mousemove_event(event);
            };
        } else if(type ==='mouseup') {
            return function(event) {
                self._mouseup_event(event);
            };
        } else if(type ==='mouseleave') {
            return function(event) {
                self._mouseleave_event(event);
            };
        }
    }

    _setup_interactions() {
        /*
            Makes parent viewport move on drag of extent rectangle.
        */
        if(this.disableInteractions) return;

        this.mouseDown = false;
        this.parentViewport.addCallback(this.id, 'mousedown', this.__get_callback('mousedown'));
        this.parentViewport.addCallback(this.id, 'mousemove', this.__get_callback('mousemove'));
        this.parentViewport.addCallback(this.id, 'mouseup', this.__get_callback('mouseup'));
        // this.parentViewport.addCallback(this.id, 'mouseleave', this.__get_callback('mouseleave'));
    }

    getPosition() {
        return this.position;
    }

    setPosition(x, y, size) {
        this.position = [x, y, size, size];
    }


    minimapScaleFun(coordinates, target, backwards) {
        /*
            Transforms coordinates to this minimap's area.
        */
        var coords_out = coordinates.slice();

        if(backwards) {

            // un-shift position
            coords_out[0] -= this.pos_abs[0];
            coords_out[1] -= this.pos_abs[1];

            var canvasSize = [this.pos_abs[2], this.pos_abs[3]];
            if(target === 'canvas') {
                coords_out[0] /= canvasSize[0];
                coords_out[1] /= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] /= canvasSize[0];
                    coords_out[3] /= canvasSize[1];
                }

            } else if(target === 'validArea') {
                coords_out[0] /= canvasSize[0];
                coords_out[1] /= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] /= canvasSize[0];
                    coords_out[3] /= canvasSize[1];
                }
            }
        } else {
            var canvasSize = [this.pos_abs[2], this.pos_abs[3]];
            if(target === 'canvas') {
                coords_out[0] *= canvasSize[0];
                coords_out[1] *= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] *= canvasSize[0];
                    coords_out[3] *= canvasSize[1];
                }
            } else if(target === 'validArea') {
                coords_out[0] *= canvasSize[0];
                coords_out[1] *= canvasSize[1];
                if(coords_out.length == 4) {
                    coords_out[2] *= canvasSize[0];
                    coords_out[3] *= canvasSize[1];
                }
            }

            // shift position
            coords_out[0] += this.pos_abs[0];
            coords_out[1] += this.pos_abs[1];

            // clamp coordinates to minimap extent
            coords_out[0] = Math.max(coords_out[0], this.pos_abs[0]);
            coords_out[1] = Math.max(coords_out[1], this.pos_abs[1]);
            if(coords_out.length == 4) {
                coords_out[0] = Math.min(coords_out[0], this.pos_abs[0]+this.pos_abs[2]-coords_out[2]);
                coords_out[1] = Math.min(coords_out[1], this.pos_abs[1]+this.pos_abs[3]-coords_out[3]);
            } else {
                coords_out[0] = Math.min(coords_out[0], this.pos_abs[0]+this.pos_abs[2]);
                coords_out[1] = Math.min(coords_out[1], this.pos_abs[1]+this.pos_abs[3]);
            }
        }
        return coords_out;
    }

    render(ctx, scaleFun) {
        if(!this.visible || this.position == null) return;
        super.render(ctx, scaleFun);

        // position of minimap on parent viewport
        this.pos_abs = scaleFun(this.position, 'canvas');

        // border and background
        ctx.fillStyle = window.styles.minimap.background.fillColor;
        ctx.strokeStyle = window.styles.minimap.background.strokeColor;
        ctx.lineWidth = window.styles.minimap.background.lineWidth;
        ctx.setLineDash(window.styles.minimap.background.lineDash);
        roundRect(ctx, this.pos_abs[0] - 2, this.pos_abs[1] - 2,
            this.pos_abs[2] + 4, this.pos_abs[3] + 4,
            5, true, true);

        // elements
        for(var e=0; e<this.parentViewport.renderStack.length; e++) {

            //TODO: dirty hack to avoid rendering HoverTextElement instances and resize handles
            if(this.parentViewport.renderStack[e].hasOwnProperty('text') ||
                this.parentViewport.renderStack[e] instanceof ElementGroup) continue;
            this.parentViewport.renderStack[e].render(ctx, (this.minimapScaleFun).bind(this));
        }

        // current extent of parent viewport
        var extent_parent = this.minimapScaleFun(this.parentViewport.getViewport(), 'canvas');
        ctx.fillStyle = window.styles.minimap.viewport.fillColor;
        ctx.strokeStyle = window.styles.minimap.viewport.strokeColor;
        ctx.lineWidth = window.styles.minimap.viewport.lineWidth;
        ctx.setLineDash(window.styles.minimap.viewport.lineDash);
        ctx.fillRect(extent_parent[0], extent_parent[1],
            extent_parent[2], extent_parent[3]);


        // another outlined border for aesthetics
        ctx.strokeStyle = window.styles.minimap.background.strokeColor;
        ctx.lineWidth = window.styles.minimap.background.lineWidth;
        ctx.setLineDash(window.styles.minimap.background.lineDash);
        roundRect(ctx, this.pos_abs[0] - ctx.lineWidth/2, this.pos_abs[1] - ctx.lineWidth/2,
            this.pos_abs[2] + ctx.lineWidth, this.pos_abs[3] + ctx.lineWidth,
            5, false, true);
    }
}



class SegmentationElement extends AbstractRenderElement {

    constructor(id, imageData, width, height, zIndex, disableInteractions) {
        super(id, null, zIndex, disableInteractions);
        this._create_canvas(imageData, width, height);
    }

    _create_canvas(imageData, width, height) {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.ctx.imageSmoothingEnabled = false;
        if(width && height) {
            this.setSize([width, height]);
        }
        
        // add image data to canvas if available
        if(imageData != undefined && imageData != null) {
            this._parse_map(window.base64ToBuffer(imageData));
        }
    }

    setSize(size) {
        this.canvas.width = size[0];
        this.canvas.height = size[1];
    }


    /* export and conversion functions */
    getGeometry() {
        return {
            'type': 'segmentationMask',
            'segmentationMask': window.bufferToBase64(this._export_map()),
            'width': this.canvas.width,
            'height': this.canvas.height
        };
    }

    _parse_map(indexedData) {
        /*
            Receives an array of pixel values corresponding to class
            indices. Fills the canvas with RGB values of the respective
            class, or zeros if no class match could be found.
        */
        
        // get current canvas pixel values for a quick template
        var pixels = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        var data = pixels.data;

        // iterate over index data and assign
        var nothing = [0,0,0];
        var offset = 0;
        var color = nothing;
        var alpha = 0;
        for(var i=0; i<indexedData.length; i++) {
            // find label class color at position
            var lc = window.labelClassHandler.getByIndex(indexedData[i]);
            if(lc) {
                color = lc.colorValues;
                alpha = 255;
            } else {
                color = nothing;
                alpha = 0;
            }
            data[offset] = color[0];
            data[offset+1] = color[1];
            data[offset+2] = color[2];
            data[offset+3] = alpha;
            offset += 4;
        }
        this.ctx.putImageData(new ImageData(data, this.canvas.width, this.canvas.height), 0, 0);
    }

    _export_map() {
        /*
            Parses the RGBA canvas map and returns an array with
            pixel values that correspond to the class index at the given position,
            indicated by the canvas color.
            Pixels receive value -1 if no class match could be found.
        */

        // get pixel values
        var pixels = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        var data = pixels.data;

        // convert to labelclass idx
        var indexedData = [];
        for(var i=0; i<data.length; i+=4) {
            // find label class at position
            var lc = window.labelClassHandler.getByColor(data.slice(i,i+3));
            indexedData.push(lc === null || lc === undefined ? -1 : lc.index);
        }
        return new Uint8Array(indexedData);
    }


    /* painting functions */
    _clear_circle(x, y, radius) {
        this.ctx.beginPath();
        this.ctx.globalCompositeOperation = 'destination-out'
        this.ctx.arc(x, y, radius, 0, Math.PI*2, true);
        this.ctx.fill();
        this.ctx.closePath();
        this.ctx.globalCompositeOperation = 'source-over';
    }

    paint(coords, color, brushType, brushSize) {
        this.ctx.fillStyle = color;
        if(brushType === 'rectangle') {
            this.ctx.fillRect(coords[0] - brushSize/2, coords[1] - brushSize/2,
                brushSize, brushSize);
        } else if(brushType === 'circle') {
            this.ctx.beginPath();
            this.ctx.arc(coords[0], coords[1], brushSize/2, 0, 2*Math.PI);
            this.ctx.fill();
            this.ctx.closePath();
        }
    }

    clear(coords, brushType, brushSize) {
        if(brushType === 'rectangle') {
            this.ctx.clearRect(coords[0] - brushSize/2, coords[1] - brushSize/2,
                brushSize, brushSize);
        } else if(brushType === 'circle') {
            this._clear_circle(coords[0], coords[1], brushSize/2)
        }
    }

    clearAll() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    render(ctx, scaleFun) {
        if(!this.visible) return;
        super.render(ctx, scaleFun);
        
        var targetCoords = scaleFun([0,0,1,1], 'validArea');
        
        // draw canvas as an image
        ctx.drawImage(
            this.canvas,
            targetCoords[0], targetCoords[1],
            targetCoords[2], targetCoords[3]
        )
    }
}