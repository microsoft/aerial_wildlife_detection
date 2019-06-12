class AbstractRenderElement {

    constructor(zIndex) {
        this.zIndex = (zIndex == null? 0 : zIndex);
    }

    setProperty(propertyName, value) {
        this[propertyName] = value;
    }

    zIndex() {
        return this.zIndex;
    }

    render(ctx, viewport, scaleFun) {
        throw Error('Not implemented.');
    }
}


class ElementGroup extends AbstractRenderElement {

    constructor(elements, zIndex) {
        super(zIndex);
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

    render(ctx, viewport, scaleFun) {
        for(var e=0; e<this.elements.length; e++) {
            this.elements[e].render(ctx, viewport, scaleFun);
        }
    }
}


class ImageElement extends AbstractRenderElement {

    constructor(viewport, imageURI, width, height, zIndex) {
        super(zIndex);
        this.viewport = viewport;
        this.imageURI = imageURI;
        this.width = width;
        this.height = height;
        this._create_image();
    }

    _create_image() {
        this.image = new Image();
        this.image.width = this.width;
        this.image.height = this.height;
        var self = this;
        this.image.onload = function() {
            self.viewport.render();
        };
        this.image.src = this.imageURI;
    }

    render(ctx, viewport, scaleFun) {
        var targetCoords = scaleFun(viewport);
        ctx.drawImage(this.image, targetCoords[0], targetCoords[1],
            targetCoords[0]+targetCoords[2],
            targetCoords[1]+targetCoords[3]);
    }
}



class HoverTextElement extends AbstractRenderElement {

    constructor(hoverText, position, zIndex) {
        super(zIndex);
        this.text = hoverText;
        this.position = position;
    }

    render(ctx, viewport, scaleFun) {
        if(this.text == null) return;
        var hoverPos = scaleFun(this.position);
        var dimensions = ctx.measureText(this.text);
        dimensions.height = window.styles.hoverText.box.height;
        var offsetH = window.styles.hoverText.offsetH;
        ctx.fillStyle = window.styles.hoverText.box.fill;
        ctx.fillRect(offsetH+hoverPos[0]-2, hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
        ctx.strokeStyle = window.styles.hoverText.box.stroke.color;
        ctx.lineWidth = window.styles.hoverText.box.stroke.lineWidth;
        ctx.strokeRect(offsetH+hoverPos[0]-2, hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
        ctx.fillStyle = window.styles.hoverText.text.color;
        ctx.font = window.styles.hoverText.text.font;
        ctx.fillText(this.text, offsetH+hoverPos[0], hoverPos[1]);
    }
}



class PointElement extends AbstractRenderElement {

    constructor(x, y, color, size, zIndex) {
        super(zIndex);
        this.x = x;
        this.y = y;
        this.color = color;
        this.size = size;
    }

    render(ctx, viewport, scaleFun) {
        if(this.x == null || this.y == null) return;
        var coords = scaleFun([this.x, this.y]);
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(coords[0], coords[1], this.size, 0, 2*Math.PI);
        ctx.fill();
        ctx.closePath();
    }
}



class LineElement extends AbstractRenderElement {

    constructor(startX, startY, endX, endY, strokeColor, lineWidth, lineDash, zIndex) {
        super(zIndex);
        this.startX = startX;
        this.startY = startY;
        this.endX = endX;
        this.endY = endY;
        this.strokeColor = strokeColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);
    }

    render(ctx, viewport, scaleFun) {
        if(this.startX == null || this.startY == null ||
            this.endX == null || this.endY == null)
            return;
        var startPos = scaleFun([this.startX, this.startY]);
        var endPos = scaleFun([this.endX, this.endY]);
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


class RectangleElement extends AbstractRenderElement {

    constructor(x, y, width, height, fillColor, strokeColor, lineWidth, lineDash, zIndex) {
        super(zIndex);
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.fillColor = fillColor;
        this.color = strokeColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);
    }

    render(ctx, viewport, scaleFun) {
        if(this.x == null || this.y == null) return;
        var coords = scaleFun([this.x, this.y, this.width, this.height]);
        if(this.fillColor != null) {
            ctx.fillStyle = this.fillColor;
            ctx.fillRect(coords[0] - coords[2]/2, coords[1] - coords[3]/2, coords[2], coords[3]);
        }
        if(this.color != null) {
            ctx.strokeStyle = this.color;
            ctx.lineWidth = this.lineWidth;
            ctx.setLineDash(this.lineDash);
            ctx.beginPath();
            ctx.strokeRect(coords[0] - coords[2]/2, coords[1] - coords[3]/2, coords[2], coords[3]);
            ctx.closePath();
        }
    }
}



class BorderStrokeElement extends AbstractRenderElement {
    /*
        Draws a border around the viewport.
        Specifically intended for classification tasks.
    */
    constructor(strokeColor, lineWidth, lineDash, zIndex) {
        super(zIndex);
        this.color = strokeColor;
        this.lineWidth = lineWidth;
        this.lineDash = (lineDash == null? [] : lineDash);
    }

    render(ctx, viewport, scaleFun) {
        if(this.color == null) return;
        var coords = scaleFun(viewport);
        ctx.strokeStyle = this.color;
        ctx.lineWidth = this.lineWidth;
        ctx.setLineDash(this.lineDash);
        ctx.beginPath();
        ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
        ctx.closePath();
    }
}