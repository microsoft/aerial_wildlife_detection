/*
 * 2020 Benjamin Kellenberger
 */


const NODE_NAMES = {
    'train': 'Train',
    'inference': 'Inference',
    'repeater': 'Repeat',
    'connector': 'Connect'
}

class AbstractNode {
    constructor(parent, params, nodeType) {
        if(params === undefined || params === null) {
            params = {};
        }
        if(params.hasOwnProperty('id')) {
            this.id = params['id'];
        } else {
            this.id = parent.newID();
        }
        this.params = params;
        this.parent = parent;
        this.nodeType = nodeType;
        this.active = false;
        this.connectionLines = [undefined, undefined];  // incoming, outgoing
        this.repeaterLines = [undefined, undefined];    // same principle, special hooks for repeaters
    }

    _check_params(params, defaults) {
        this.params = params;
        if(this.params === null || this.params === undefined) {
            this.params = defaults;
        } else {
            for(var key in defaults) {
                if(!this.params.hasOwnProperty(key)) {
                    this.params[key] = defaults[key];
                }
            }
        }

        // remove duplicates
        delete this.params['id'];
        delete this.params['type'];
    }

    _setup_markup(position) {
        this.markup = $('<div class="node" id="'+this.id+'"></div>');
        this.markup.addClass('node-' + this.nodeType);
        this.setPosition(position);
    }

    _remove() {
        this.parent.removeNode(this.id);
    }

    remove() {
        this.unhookConnectingLine(false, true, false);
        this.unhookConnectingLine(true, true, false);
        this.unhookConnectingLine(false, true, true);
        this.unhookConnectingLine(true, true, true);
        $(this.markup).remove();
    }

    setActive(active) {
        this.active = active;
        if(this.active) {
            this.markup.addClass('active-workflow-node');
        } else {
            this.markup.removeClass('active-workflow-node');
        }
    }

    distanceToMarkup(point) {
        var ext = this.getExtent();
        if(this.markupContainsPoint(point)) {
            return 0;
        } else {
            return Math.sqrt(
                Math.pow(
                    Math.min(
                        Math.abs(ext[0] - point[0]),
                        Math.abs(ext[0]+ext[2] - point[0])
                    ), 2
                ) +
                Math.pow(
                    Math.min(
                        Math.abs(ext[1] - point[1]),
                        Math.abs(ext[1]+ext[3] - point[1])
                    ), 2
                )
            );
        }
    }

    markupContainsPoint(point) {
        var ext = this.getExtent();
        return (point[0] >= ext[0] && point[0] <= (ext[0]+ext[2])) &&
                (point[1] >= ext[1] && point[1] <= (ext[1]+ext[3]));
    }

    getConnectingLine(incoming, repeater) {
        var index = (incoming ? 0 : 1);
        if(repeater) {
            return this.repeaterLines[index];
        } else {
            return this.connectionLines[index];
        }
    }

    setConnectingLine(line, incoming, repeater) {
        var index = (incoming ? 0 : 1);
        if(repeater) {
            if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined)
                this.repeaterLines[index].setConnectingNode(null, !incoming, true);
            this.repeaterLines[index] = line;
            if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined)
                this.repeaterLines[index].setConnectingNode(this, !incoming, true);
        } else {
            if(this.connectionLines[index] !== null && this.connectionLines[index] !== undefined)
                this.connectionLines[index].setConnectingNode(null, !incoming, false);
            this.connectionLines[index] = line;
            if(this.connectionLines[index] !== null && this.connectionLines[index] !== undefined)
                this.connectionLines[index].setConnectingNode(this, !incoming, false);
        }
    }

    unhookConnectingLine(incoming, remove, repeater) {
        var index = (incoming ? 0 : 1);
        if(repeater) {
            if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined) {
                this.repeaterLines[index].setConnectingNode(null, index);   // no index flipping; repeater nodes' flow is inverted
                if(remove) {
                    this.repeaterLines[index].remove();
                }
            }
        } else {
            if(this.connectionLines[index] !== null && this.connectionLines[index] !== undefined) {
                this.connectionLines[index].setConnectingNode(null, (1-index));
                if(remove) {
                    this.connectionLines[index].remove();
                }
            }
        }
    }

    getMarkup() {
        return this.markup[0];
    }

    getExtent() {
        return [
            parseInt(this.markup.css('left')),
            parseInt(this.markup.css('top')),
            parseInt(this.markup.css('width')),
            parseInt(this.markup.css('height'))
        ];
    }

    setPosition(position) {
        this.position = position;
        if(Array.isArray(position) && 
            typeof(position[0]) === 'number' && typeof(position[1]) === 'number') {
            this.markup.css('left', position[0]);
            this.markup.css('top', position[1]);
            this.markup.css('display', 'block');
        } else {
            this.markup.css('display', 'none');
        }
        this.notifyPositionChange();
    }

    move(shiftVector) {
        this.markup.css('left', parseInt(this.markup.css('left')) + shiftVector[0]);
        this.markup.css('top', parseInt(this.markup.css('top')) + shiftVector[1]);
        this.notifyPositionChange();
    }

    notifyPositionChange() {
        for(var i=0; i<this.connectionLines.length; i++) {
            if(this.connectionLines[i] !== undefined && this.connectionLines[i] !== null) {
                this.connectionLines[i].notifyPositionChange();
            }
        }
        for(var i=0; i<this.repeaterLines.length; i++) {
            if(this.repeaterLines[i] !== undefined && this.repeaterLines[i] !== null) {
                this.repeaterLines[i].notifyPositionChange();
            }
        }
    }

    getPreviousElement(repeater) {
        if(repeater) {
            if(this.repeaterLines[0] !== undefined && this.repeaterLines[0] !== null) {
                return this.repeaterLines[0].node_1;
            } else {
                return undefined;
            }
        } else {
            if(this.connectionLines[0] !== undefined && this.connectionLines[0] !== null) {
                return this.connectionLines[0].node_1;
            } else {
                return undefined;
            }
        }
    }

    getNextElement(repeater) {
        if(repeater) {
            if(this.repeaterLines[1] !== undefined && this.repeaterLines[1] !== null) {
                return this.repeaterLines[1].node_2;
            } else {
                return undefined;
            }
        } else {
            if(this.connectionLines[1] !== undefined && this.connectionLines[1] !== null) {
                return this.connectionLines[1].node_2;
            } else {
                return undefined;
            }
        }
    }

    toJSON() {
        return {
            id: this.id,
            type: this.nodeType,
            kwargs: this.params,
            extent: this.getExtent()
        }
    }
}


class DummyNode extends AbstractNode {
    /**
     * Used e.g. for re-positioning or bending lines.
     * Can either show nothing at all or a move handle.
     */
    DEFAULT_PARAMS = {
        show_handle: true
    }
    constructor(parent, params, position) {
        super(parent, params, 'dummy');
        super._check_params(params['kwargs'], this.DEFAULT_PARAMS);
        this.position = position;
        this._setup_markup(position);
    }

    _setup_markup(position) {
        this.markup = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        this.setPosition(position);
        this.markup.setAttribute('width', 5);
        this.markup.setAttribute('height', 5);
        this.markup.setAttribute('stroke', 'black');
        this.markup.setAttribute('stroke-width', 2);
        this.markup.setAttribute('fill', 'white');
        if(!this.params['show_handle']) {
            this.markup.setAttribute('visibility', 'hidden');
        }
    }

    getMarkup() {
        return this.markup;
    }

    getExtent() {
        return [parseInt(this.markup.getAttribute('x')), parseInt(this.markup.getAttribute('y')), 0, 0];
    }

    setPosition(position) {
        this.position = position;
        if(Array.isArray(position) &&
            typeof(position[0]) === 'number' && typeof(position[1]) === 'number') {
            this.markup.setAttribute('x', this.position[0]);
            this.markup.setAttribute('y', this.position[1]);
            if(this.params['show_handle']) {
                this.markup.setAttribute('visibility', 'visible');
            } else {
                this.markup.setAttribute('visibility', 'hidden');
            }
        } else {
            this.markup.setAttribute('visibility', 'hidden');
        }
    }

    move(shiftVector) {
        var pos = this.getExtent();
        this.markup.setAttribute('x', pos[0] + shiftVector[0]);
        this.markup.setAttribute('y', pos[1] + shiftVector[1]);
        this.notifyPositionChange();
    }

    toJSON() {
        return null;
    }
}


class ConnectionNode extends AbstractNode {
    DEFAULT_PARAMS = {
        is_start_node: false
    }
    constructor(parent, params, position) {
        super(parent, params, 'connector');
        this._setup_markup(position);
        this._check_params(params, this.DEFAULT_PARAMS);
    }

    _setup_markup(position) {
        super._setup_markup(position);
        if(this.params['is_start_node']) {
            this.markup.append('<div>Start</div>');
        }
    }

    setConnectingLine(line, incoming, repeater) {
        if(this.params['is_start_node'] && incoming && !repeater) {
            // regular incoming lines are not allowed for starting node
            return;
        } else {
            super.setConnectingLine(line, incoming, repeater);
        }
    }
}



class DefaultNode extends AbstractNode {
    constructor(parent, params, nodeType) {
        super(parent, params, nodeType);
    }

    _setup_markup(position) {
        super._setup_markup(position);
        var self = this;

        // delete node button
        var delBtn = $('<button class="btn btn-sm btn-secondary">X</button>');
        delBtn.click(function() {
            self._remove();
        });
        this.markup.append(delBtn);

        // node type name
        this.nodeTitleMarkup = $('<span class="node-title">'+NODE_NAMES[this.nodeType]+'</span>');
        this.markup.append(this.nodeTitleMarkup);
    }
}


class TrainNode extends DefaultNode {
    DEFAULT_PARAMS = {
        'min_timestamp': 'lastState',
        'min_anno_per_image': 0,
        'include_golden_questions': false,
        'max_num_images': 0,
        'max_num_workers': 0
    }

    constructor(parent, params, position) {
        super(parent, params, 'train');
        super._check_params(params['kwargs'], this.DEFAULT_PARAMS);
        this._setup_markup(position);
    }

    _setup_markup(position) {
        super._setup_markup(position);
        var self = this;

        this.propertiesMarkup = $('<div style="display:none"></div>');
        this.markup.append(this.propertiesMarkup);
        this.nodeTitleMarkup.click(function() {
            if(self.active) {
                self.propertiesMarkup.slideToggle(400, function() {
                    self.notifyPositionChange();
                });
            }
        });

        //TODO: set defaults correctly...
        //TODO 2: add date & time chooser for min. timestamp
        //TODO 3: figure out if min. timestamp is for model or images...

        // minimum timestamp
        var minTmarkup = $('<div></div>');
        minTmarkup.append('<div>Images last viewed:</div>');
        this.minTgr = $('<buttonGroup></buttonGroup>');
        this.minTgr.append($('<input type="radio" name="min-timestamp" id="'+this.id+'-minT-latest" value="lastState" checked="checked" />' +
                            '<label for="'+this.id+'-minT-latest">Latest</label>'));
        this.minTgr.append($('<br />'));
        this.minTgr.append($('<input type="radio" name="min-timestamp" id="'+this.id+'-minT-latest" value="timestamp" />' +
                            '<label for="'+this.id+'-minT-latest">From date on:</label>'));
        minTmarkup.append(this.minTgr);
        this.propertiesMarkup.append(minTmarkup);

        // golden question images
        this.gqchck = $('<input type="checkbox" id="gqchck_' + this.id + '" />');
        var chckbxMarkup = $('<div></div>');
        chckbxMarkup.append(this.gqchck);
        chckbxMarkup.append($('<label for="gqchck_' + this.id + '">Include golden questions</label>'));
        this.propertiesMarkup.append(chckbxMarkup);

        var optionsTable = $('<table></table>');
        this.propertiesMarkup.append(optionsTable);

        // minimum number of annotations
        var minNumAnnoMarkup = $('<tr></tr>');
        minNumAnnoMarkup.append('<td>Minimum no. annotations per image:</td>');
        var tdNumAnno = $('<td></td>');
        this.minNumAnno = $('<input type="number" value="0" min="0" max="1024" style="width:65px" />');
        tdNumAnno.append(this.minNumAnno);
        minNumAnnoMarkup.append(tdNumAnno);
        optionsTable.append(minNumAnnoMarkup);

        // maximum number of images
        var maxNumImgsMarkup = $('<tr></tr>');
        maxNumImgsMarkup.append($('<td>Number of images (0 = unlimited):</td>'));
        var tdNumImg = $('<td></td>');
        this.maxNumImgs = $('<input type="number" value="0" min="0" max="1000000000" style="width:65px" />');
        tdNumImg.append(this.maxNumImgs);
        maxNumImgsMarkup.append(tdNumImg);
        optionsTable.append(maxNumImgsMarkup);

        // number of workers (TODO: query available number from server first)
        var maxNumWorkersMarkup = $('<tr></tr>');
        maxNumWorkersMarkup.append($('<td>Number of workers (0 = unlimited):</td>'));
        var tdNumWorkers = $('<td></td>');
        this.maxNumWorkers = $('<input type="number" value="0" min="0" max="1000000000" style="width:65px" />');
        tdNumWorkers.append(this.maxNumWorkers);
        maxNumWorkersMarkup.append(tdNumWorkers);
        optionsTable.append(maxNumWorkersMarkup);
    }

    toJSON() {
        // collect parameters first
        var timestampSel = this.minTgr.find('input:radio[name="min-timestamp"]:checked').val();
        if(timestampSel === 'timestamp') {
            // get from date instead
            timestampSel = new Date();  //TODO
        }
        this.params['min_timestamp'] = timestampSel;
        this.params['min_anno_per_image'] = this.minNumAnno.val();
        this.params['include_golden_questions'] = this.gqchck.prop('checked');
        this.params['max_num_images'] = this.maxNumImgs.val();
        this.params['max_num_workers'] = this.maxNumWorkers.val();
        return super.toJSON();
    }
}


class InferenceNode extends DefaultNode {
    DEFAULT_PARAMS = {
        'force_unlabeled': false,       //TODO
        'golden_questions_only': false, //TODO
        'max_num_images': 0,
        'max_num_workers': 0
    }

    constructor(parent, params, position) {
        super(parent, params, 'inference');
        super._check_params(params['kwargs'], this.DEFAULT_PARAMS);
        this._setup_markup(position);
    }

    _setup_markup(position) {
        super._setup_markup(position);
        var self = this;

        this.propertiesMarkup = $('<div style="display:none"></div>');
        this.markup.append(this.propertiesMarkup);
        this.nodeTitleMarkup.click(function() {
            if(self.active) {
                self.propertiesMarkup.slideToggle(400, function() {
                    self.notifyPositionChange();
                });
            }
        });

        // force unlabeled images
        this.unlabChck = $('<input type="checkbox" id="fUnlabeled_' + this.id + '" />');
        this.unlabChck.prop('checked', this.params['force_unlabeled']);
        var chckbxMarkup_u = $('<div></div>');
        chckbxMarkup_u.append(this.unlabChck);
        chckbxMarkup_u.append($('<label for="fUnlabeled_' + this.id + '">Limit to unlabeled images</label>'));
        this.propertiesMarkup.append(chckbxMarkup_u);

        // golden question images
        this.gqchck = $('<input type="checkbox" id="gqchck_' + this.id + '" />');
        this.gqchck.prop('checked', this.params['golden_questions_only']);
        var chckbxMarkup_gq = $('<div></div>');
        chckbxMarkup_gq.append(this.gqchck);
        chckbxMarkup_gq.append($('<label for="gqchck_' + this.id + '">Limit to golden questions</label>'));
        this.propertiesMarkup.append(chckbxMarkup_gq);

        var optionsTable = $('<table></table>');
        this.propertiesMarkup.append(optionsTable);

        // maximum number of images
        var maxNumImgsMarkup = $('<tr></tr>');
        maxNumImgsMarkup.append($('<td>Number of images (0 = unlimited):</td>'));
        this.maxNumImgs = $('<td><input type="number" value="0" min="0" max="1000000000" style="width:65px" /></td>');
        maxNumImgsMarkup.append(this.maxNumImgs);
        optionsTable.append(maxNumImgsMarkup);

        // number of workers (TODO: query available number from server first)
        var maxNumWorkersMarkup = $('<tr></tr>');
        maxNumWorkersMarkup.append($('<td>Number of workers (0 = unlimited):</td>'));
        this.maxNumWorkers = $('<td><input type="number" value="0" min="0" max="1000000000" style="width:65px" /></td>');
        maxNumWorkersMarkup.append(this.maxNumWorkers);
        optionsTable.append(maxNumWorkersMarkup);
    }

    toJSON() {
        // collect parameters first
        this.params['force_unlabeled'] = this.unlabChck.prop('checked');
        this.params['golden_questions_only'] = this.gqchck.prop('checked');
        this.params['max_num_images'] = this.maxNumImgs.val();
        this.params['max_num_workers'] = this.maxNumWorkers.val();
        return super.toJSON();
    }
}


class RepeaterNode extends AbstractNode {
    DEFAULT_PARAMS = {
        num_repetitions: 1,
        max_num_repetitions: 500
    }

    constructor(parent, params, position) {
        super(parent, params, 'repeater');
        super._check_params(params['kwargs'], this.DEFAULT_PARAMS);
        this._setup_markup(position);
    }

    _setup_markup(position) {
        super._setup_markup(position);
        var self = this;

        // delete node
        var delBtn = $('<button class="btn btn-sm btn-secondary" style="float:left;margin-right:10px">X</button>');
        delBtn.click(function() {
            self._remove();
        });
        this.markup.append(delBtn);

        var wrapper = $('<div></div>');
        wrapper.append($('<span>Repeat </span>'));
        this.numRepCounter = $('<input type="number" min="1" max="' + 
                                this.params['max_num_repetitions'] +
                                '" value="' + this.params['num_repetitions'] + '" />');
        wrapper.append(this.numRepCounter);
        wrapper.append($('<span> times</span>'));
        this.markup.append(wrapper);
    }

    getConnectingLine(incoming, repeater) {
        var index = (incoming ? 0 : 1);
        return this.repeaterLines[index];
    }

    setConnectingLine(line, incoming, repeater) {
        var index = (incoming ? 0 : 1);
        if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined)
            this.repeaterLines[index].setConnectingNode(null, !incoming, true);
        this.repeaterLines[index] = line;
        if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined)
            this.repeaterLines[index].setConnectingNode(this, !incoming, true);
    }

    unhookConnectingLine(incoming, remove, repeater) {
        var index = (incoming ? 0 : 1);
        if(this.repeaterLines[index] !== null && this.repeaterLines[index] !== undefined) {
            this.repeaterLines[index].setConnectingNode(null, (1-index), true);
            if(remove) {
                this.repeaterLines[index].remove();
            }
        }
    }

    toJSON() {
        var params = super.toJSON();
        try {
            params['start_node'] = this.getPreviousElement(true).id
            params['end_node'] = this.getNextElement(true).id
        } catch {
            return null;
        }
        params['kwargs']['num_repetitions'] = parseInt(this.numRepCounter.val());
        return params;
    }
}


class ConnectionLine {
    constructor(id, parent, node_1, node_2, is_repeater) {
        this.id = id;
        this.parent = parent;
        this.active = false;
        this.node_1 = node_1;
        this.node_2 = node_2;
        this.is_repeater = is_repeater;
        this.repeater_outgoing = (this.node_1 instanceof RepeaterNode);
        this.position = [0,0,0,0];
        this._setup_markup();
        if(this.node_1 !== undefined && this.node_1 !== null) {
            this.node_1.setConnectingLine(this, false, this.is_repeater);
        }
        if(this.node_2 !== undefined && this.node_2 !== null) {
            this.node_2.setConnectingLine(this, true, this.is_repeater);
        }
    }

    _setup_markup() {
        // we use a polyline to be able to draw arrows in the middle
        this.markup = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        this.markup.setAttribute('points', '0,0 0,0 0,0');
        this.markup.setAttribute('stroke', 'black');
        this.markup.setAttribute('stroke-width', 2);
        if(this.is_repeater) {
            this.markup.setAttribute('stroke-dasharray', 2);
        }
        this.markup.setAttribute('marker-mid', 'url(#arrow)');

        this._update_line();
    }

    _update_line() {
        if(this.node_1 !== undefined && this.node_1 !== null &&
            this.node_2 !== undefined && this.node_2 !== null) {
            
            var extent_1 = this.node_1.getExtent();
            var startPos = [extent_1[0] + extent_1[2]/2, extent_1[1] + extent_1[3]/2];

            var extent_2 = this.node_2.getExtent();
            var endPos = [extent_2[0] + extent_2[2]/2, extent_2[1] + extent_2[3]/2];

            // shift positions if this is a repeater line
            if(this.is_repeater) {
                if(this.repeater_outgoing) {
                    startPos = [startPos[0] - extent_1[2]/6, startPos[1] - extent_1[3]/6];
                    endPos = [endPos[0] - extent_2[2]/6, endPos[1] - extent_2[3]/6];
                } else {
                    startPos = [startPos[0] + extent_1[2]/6, startPos[1] + extent_1[3]/6];
                    endPos = [endPos[0] + extent_2[2]/6, endPos[1] + extent_2[3]/6];
                }
            }
            this.position = startPos.concat(endPos);
            this.markup.setAttribute('points',
                startPos[0] + ',' + startPos[1] + ' ' +
                (startPos[0]+endPos[0])/2 + ',' + (startPos[1]+endPos[1])/2 + ' ' +
                endPos[0] + ',' + endPos[1]
            );
            if(this.active) {
                // this.markup.setAttribute('stroke', '#ebc246');
                this.markup.setAttribute('stroke-width', 4);
            } else {
                this.markup.setAttribute('stroke', 'black');
                this.markup.setAttribute('stroke-width', 2);
            }
            this.markup.setAttribute('visibility', 'visible');

        } else {
            // at least one node is missing; hide line
            this.markup.setAttribute('visibility', 'hidden');
        }
    }

    getMarkup() {
        return this.markup;
    }

    setActive(active) {
        this.active = active;
        this._update_line();
    }

    getConnectingNode(incoming) {
        return (incoming ? this.node_1 : this.node_2);
    }

    setConnectingNode(node, incoming) {
        if(incoming) {
            this.node_1 = node;
        } else {
            this.node_2 = node;
        }
        this._update_line();
    }

    notifyPositionChange() {
        this._update_line();
    }

    remove() {
        // notify all the nodes about removal
        if(this.node_1 !== undefined && this.node_1 !== null)
            this.node_1.setConnectingLine(undefined, false);
        if(this.node_2 !== undefined && this.node_2 !== null)
            this.node_2.setConnectingLine(undefined, true);
        this.markup.remove();
    }

    distanceToMarkup(point) {
        var pos_1 = [this.position[0], this.position[1]];
        var pos_2 = [this.position[2], this.position[3]];

        // check if point within MBR
        if(point[0] >= Math.min(pos_1[0], pos_2[0]) && point[0] < Math.max(pos_1[0], pos_2[0]) &&
            point[1] >= Math.min(pos_1[1], pos_2[1]) && point[1] < Math.max(pos_1[1], pos_2[1])) {

            // distance to line
            var a = Math.abs((point[0] * (pos_2[1]-pos_1[1])) -
                    point[1] * (pos_2[0]-pos_1[0]) +
                    pos_2[0]*pos_1[1] - pos_2[1]*pos_1[0]);
            var b = Math.sqrt(Math.pow(pos_2[1]-pos_1[1], 2) +
                    Math.pow(pos_2[0]-pos_1[0], 2));
            return a/b;

        } else {
            // distance to endpoints
            return Math.min(
                Math.sqrt(Math.pow(pos_1[0]-point[0], 2) +
                        Math.pow(pos_1[1]-point[1], 2)),
                Math.sqrt(Math.pow(pos_2[0]-point[0], 2) +
                        Math.pow(pos_2[1]-point[1], 2))
            );
        }
    }

    markupContainsPoint(point) {
        return this.distanceToMarkup(point) <= 5;
    }

    move(shiftVector) {
        // not implementable for line per se (TODO)
    }
}


class Canvas {
    constructor(domElement, type, zIndex) {
        this.type = type;
        this.zIndex = zIndex;
        this.elements = [];
        this._setup_markup(domElement);
    }

    _setup_markup(domElement) {
        if(this.type === 'dom') {
            this.canvas = $('<div class="canvas"></div>')[0];
        } else {
            this.canvas = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            this.canvas.classList.add('svg-canvas');

            // initialize arrow for lines
            var arrow = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
            arrow.setAttribute('id', 'arrow');
            arrow.setAttribute('markerWidth', 10);
            arrow.setAttribute('markerHeight', 10);
            arrow.setAttribute('refX', 0);
            arrow.setAttribute('refY', 3);
            arrow.setAttribute('orient', 'auto');
            arrow.setAttribute('markerUnits', 'strokeWidth');
            var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', 'M0,0 L0,6 L9,3 z');
            path.setAttribute('fill', 'black');
            arrow.appendChild(path);
            var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            defs.appendChild(arrow);
            this.canvas.appendChild(defs)
        }
        this.canvas.style['z-index'] = this.zIndex;
        domElement.appendChild(this.canvas);
    }

    newID() {
        var id = Math.random().toString(36).substr(2, 9);
        var idTaken = true;
        do {
            for(var i=0; i<this.elements.length; i++) {
                if(this.elements[i].id === id) {
                    id = Math.random().toString(36).substr(2, 9);
                    break;
                }
            }
            idTaken = false;
        } while(idTaken);
        return id;
    }

    clear(startingNode) {
        for(var i=0; i<this.elements.length; i++) {
            this.elements[i].remove();
            //$(this.elements[i].markup).remove();
        }
        this.elements = [];
        if(startingNode !== undefined && startingNode !== null) {
            this.addElement(startingNode);
        }
    }

    addElement(element) {
        this.elements.push(element);
        this.canvas.appendChild(element.getMarkup());
    }

    removeElement(id) {
        for(var i=0; i<this.elements.length; i++) {
            if(this.elements[i].id === id) {
                this.elements[i].remove();
                this.elements.splice(i,1);
            }
        }
    }

    getClosestElement(point) {
        var closest = null;
        var distance = 1e9;
        for(var i=this.elements.length-1; i>=0; i--) {
            var dist = this.elements[i].distanceToMarkup(point);
            if(dist < distance) {
                distance = dist;
                closest = this.elements[i];
            }
        }
        if(distance > 10) {
            closest = null;
        }
        return closest;
    }
}


class WorkflowDesigner {
    constructor(domElement, workflow) {
        this.domElement = domElement;
        this._setup_canvas(domElement);
        this._setup_callbacks(domElement);

        this.startingNode = new ConnectionNode(this, {is_start_node:true}, [20, 20]);
        this.mainCanvas.addElement(this.startingNode);

        if(Array.isArray(workflow) && workflow.length > 0) {
            this.fromJSON(workflow);
        }
    }

    _setup_canvas(domElement) {
        /**
         * We set up three layers:
         * - main HTML DOM layer
         * - bottom SVG layer (for the lines)
         */
        this.mainCanvas = new Canvas(domElement, 'dom', 1);
        this.bottomCanvas = new Canvas(domElement, 'svg', 0);
    }

    _get_last_node() {
        var lastNode = this.startingNode;
        var hasNeighbor = true;
        do {
            if(lastNode.getNextElement(false) !== undefined && lastNode.getNextElement(false) !== null) {
                lastNode = lastNode.getNextElement(false);
                if(lastNode !== undefined && lastNode !== null && lastNode.id === this.startingNode.id) {
                    // loop detected
                    return undefined;
                }
            } else {
                hasNeighbor = false;
            }
        } while(hasNeighbor);
        return lastNode;
    }

    _forms_loop(testNode, targetNode) {
        /**
         * Starts from the target node and checks its outgoing neighbors.
         * Traverses along; if any of the links is the same node as the
         * test node, the test node and target node would form a loop,
         * hence true is returned. Else returns false.
         */
        if(testNode === null || testNode === undefined || targetNode === null || targetNode === undefined) {
            return false;
        }
        var latestNode = targetNode;
        var hasNeighbor = true;
        do {
            if(latestNode.id === testNode.id) return true;
            if(latestNode.getNextElement() !== undefined && latestNode.getNextElement() !== null) {
                latestNode = latestNode.getNextElement();
            } else {
                hasNeighbor = false;
            }
        } while(hasNeighbor);
        return false;
    }

    ___get_coords(e) {
        let rect = this.domElement.getBoundingClientRect();
        let scrollOffset = $(this.mainCanvas.canvas).position();
        return [e.clientX - rect.left - scrollOffset['left'], e.clientY - rect.top - scrollOffset['top']];
    }

    __handle_mousedown(e) {
        this.mousedown = true;
        this.mousePos = this.___get_coords(e);
        this.mousedownPos = this.mousePos;

        // if nothing is active: set starting node (for potential line drawing)
        if(Object.keys(this.activeElements).length === 0) {
            this.mousedownElement = undefined;
            this.tempConnectionLine.setConnectingNode(null, true);
            for(var i=this.mainCanvas.elements.length-1; i>=0; i--) {
                var elem = this.mainCanvas.elements[i];
                if(!elem.active && elem.distanceToMarkup(this.mousePos) <= 10) {
                    // mousedown on inactive node; start drawing line
                    this.mousedownElement = elem;
                    break;
                }
            }
        }
    }

    __handle_mousemove(e) {
        if(!this.mousedown) return;
        var newPos = this.___get_coords(e);
        var coordsDiff = [newPos[0] - this.mousePos[0], newPos[1] - this.mousePos[1]];
        
        // start drawing line if mouse has moved enough
        if(this.mousedownElement !== undefined) {
            var dist = Math.sqrt(
                Math.pow(this.mousedownPos[0] - newPos[0], 2) +
                Math.pow(this.mousedownPos[1] - newPos[1], 2)
            );
            if(dist > 20) {
                // start drawing line
                this.tempEndingElement.setPosition(this.mousePos);
                this.tempConnectionLine.setConnectingNode(this.tempEndingElement, false);
                this.tempConnectionLine.setConnectingNode(this.mousedownElement, true);
            }
        }

        // // move temporary node
        // this.tempEndingElement.setPosition(newPos);

        // move active elements
        for(var key in this.activeElements) {
            this.activeElements[key].move(coordsDiff);
        }
        this.mousePos = newPos;
    }

    __handle_mouseup(e) {
        this.mousedown = false;
        this.mousePos = this.___get_coords(e);
        this.mousedownPos = [undefined, undefined];

        // check if temporary line being drawn
        if(this.tempConnectionLine.getConnectingNode(true) !== null &&
            this.mousedownElement !== undefined && this.mousedownElement !== null) {
            // connect to new node if mouseup
            for(var i=this.mainCanvas.elements.length-1; i>=0; i--) {
                var elem = this.mainCanvas.elements[i];
                if(!elem.active && elem.distanceToMarkup(this.mousePos) <= 10) {
                    if(!(elem instanceof RepeaterNode)) {
                        // check if the connection would form a loop
                        var formsLoop = this._forms_loop(this.mousedownElement, elem);
                        if(formsLoop) {
                            // add repeater instead (unless starting node)
                            if((elem.id !== this.mousedownElement.id) || !(elem instanceof ConnectionNode)) {

                                // position: calculate from connecting positions
                                let pos_a = this.mousedownElement.getExtent();
                                if(this.mousedownElement === elem) {
                                    // self-repetition; offset repeater node
                                    var newPos = [pos_a[0], Math.max(0, pos_a[1] - pos_a[3]/2 - 20)];

                                } else {
                                    let pos_b = elem.getExtent();
                                    var newPos = [  //TODO: not the best solution...
                                        Math.max(0, ((pos_a[0]+pos_a[2]/2) + (pos_b[0]+pos_b[2]/2)) / 2 - 100),
                                        Math.max(0, ((pos_a[1]+pos_a[3]/2) + (pos_b[1]+pos_b[3]/2)) / 2 - 12)
                                    ];
                                }
                                var repeater = new RepeaterNode(this, {}, newPos);
                                this.mainCanvas.addElement(repeater);
                                this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, this.mousedownElement, repeater, true));
                                this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, repeater, elem, true));
                            }
                        } else {
                            // different node; add regular line
                            var isRepeater = (this.mousedownElement instanceof RepeaterNode);
                            this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, this.mousedownElement, elem, isRepeater));
                        }

                    } else {
                        // target element is a repeater
                        this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, this.mousedownElement, elem, true));
                    }
                    break;
                }
            }
        } else {
            // check for active nodes
            if(!e.shiftKey && !e.metaKey) {
                // set everything inactive
                for(var i=0; i<this.mainCanvas.elements.length; i++) {
                    this.mainCanvas.elements[i].setActive(false);
                }
                for(var i=0; i<this.bottomCanvas.elements.length; i++) {
                    this.bottomCanvas.elements[i].setActive(false);
                }
                this.activeElements = {};
            }
            var activeMain = this.mainCanvas.getClosestElement(this.mousePos);
            if(activeMain !== null) {
                activeMain.setActive(true);
                this.activeElements[activeMain.id] = activeMain;
            } else {
                // no main element; check lines
                var activeLine = this.bottomCanvas.getClosestElement(this.mousePos);
                if(activeLine !== null) {
                    activeLine.setActive(true);
                    this.activeElements[activeLine.id] = activeLine;
                }
            }
        }
        this.mousedownElement = undefined;
        this.tempConnectionLine.setConnectingNode(null, false);
        this.tempEndingElement.setPosition([undefined, undefined]);
    }

    __handle_mouseleave(e) {
        this.mousedown = false;
        this.mousedownPos = [undefined, undefined];
        this.mousedownElement = undefined;
        this.tempConnectionLine.setConnectingNode(null, false);
        this.tempEndingElement.setPosition([undefined, undefined]);
    }

    __handle_keyup(e) {
        if(e.keyCode === 8 || e.keyCode === 46) {
            // backspace or delete; remove active elements
            for(var key in this.activeElements) {
                if(key === this.startingNode.id) continue;
                this.activeElements[key].remove();
                delete this.activeElements[key];
            }
        }
    }

    _setup_callbacks(domElement) {
        // canvas-wide variables
        this.mousedown = false;
        this.mousePos = [undefined, undefined];
        this.mousedownPos = [undefined, undefined];
        this.mousedownElement = undefined;
        this.activeElements = {};
        this.tempEndingElement = new DummyNode(this, {show_handle:false}, this.mousePos);
        this.tempConnectionLine = new ConnectionLine(this.newID(), this, null, this.tempEndingElement, false);
        this.bottomCanvas.addElement(this.tempConnectionLine);

        var self = this;
        $(domElement).on({
            'mousedown': function(e) {
                self.__handle_mousedown(e);
            },
            'mousemove': function(e) {
                self.__handle_mousemove(e);
            },
            'mouseup': function(e) {
                self.__handle_mouseup(e);
            },
            'mouseleave': function(e) {
                self.__handle_mouseleave(e);
            }
        });
        $(domElement).parent().on('keyup', function(e) {
            self.__handle_keyup(e);
        })
    }

    newID() {
        return this.mainCanvas.newID();
    }

    fromJSON(workflow) {
        // clear current nodes first
        this.clear(this.startingNode);

        if(!workflow.hasOwnProperty('tasks') || !Array.isArray(workflow['tasks'])) return;

        // add nodes
        for(var i=0; i<workflow['tasks'].length; i++) {
            this.addNode(workflow['tasks'][i]);
        }

        // add repeaters
        if(workflow.hasOwnProperty('repeaters')) {
            for(var key in workflow['repeaters']) {
                var repeaterSpec = workflow['repeaters'][key];
                this.addNode(repeaterSpec);
            }
        }
    }

    addNode(params) {
        if(typeof(params) === 'string') {
            var type = params;
            var nodeParams = {};
        } else if(params.hasOwnProperty('type')) {
            var type = params['type'];
            var nodeParams = params;
        } else {
            throw Error('Unrecognizable node type.');
        }

        // add id if not present
        if(!(nodeParams.hasOwnProperty('id'))) {
            nodeParams['id'] = this.newID();
        }

        // get position
        if(nodeParams.hasOwnProperty('extent') && Array.isArray(nodeParams['extent'])) {
            var position = nodeParams['extent'];
            var positionSpecified = true;
        } else {
            var latestNode = this._get_last_node();
            var position = latestNode.getExtent(); 
            position = [position[0]+position[2]+50, position[1]+position[3]+50]; // shift from previous element
            var positionSpecified = false;
        }
        if(type === 'train') {
            var node = new TrainNode(this, nodeParams, position);
        } else if(type === 'inference') {
            var node = new InferenceNode(this, nodeParams, position);
        } else if(type === 'repeater') {
            var node = new RepeaterNode(this, nodeParams, position);
        } else if(type === 'connector') {
            if(typeof(nodeParams) === 'object') {
                nodeParams['is_start_node'] = false;
            }
            var node = new ConnectionNode(this, nodeParams, position);
        }
        this.mainCanvas.addElement(node);

        // connect by line to the latest node
        if(!(node instanceof RepeaterNode)) {
            var line = new ConnectionLine(this.newID(), this, latestNode, node, false);
            this.bottomCanvas.addElement(line);
        } else {
            // check if repeater node has connections
            var startNodeID = nodeParams['start_node'];
            var endNodeID = nodeParams['end_node'];
            if(typeof(startNodeID) === 'string' && typeof(endNodeID) === 'string') {
                var startNode = undefined;
                var endNode = undefined;
                for(var n=0; n<this.mainCanvas.elements.length; n++) {
                    var node = this.mainCanvas.elements[n];
                    if(node.id === startNodeID) {
                        startNode = node;
                    }
                    if(node.id === endNodeID) {
                        endNode = node;
                    }
                }
                if(startNode !== undefined && endNode !== undefined) {
                    this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, startNode, node, true));
                    this.bottomCanvas.addElement(new ConnectionLine(this.newID(), this, node, endNode, true));

                    // also update position of repeater node (unless explicitly specified)
                    if(!positionSpecified) {
                        var pos_a = startNode.getExtent();
                        var pos_b = endNode.getExtent();
                        position = [  //TODO: not the best solution...
                            Math.max(0, ((pos_a[0]+pos_a[2]/2) + (pos_b[0]+pos_b[2]/2)) / 2 + 150),
                            Math.max(0, ((pos_a[1]+pos_a[3]/2) + (pos_b[1]+pos_b[3]/2)) / 2 - 80)
                        ];
                        node.setPosition(position);
                    }
                }
            }
        }

        return node;
    }

    removeNode(nodeID) {
        // find node
        var nodeIndex = -1;
        for(var i=0; i<this.mainCanvas.elements.length; i++) {
            if(this.mainCanvas.elements[i].id === nodeID) {
                nodeIndex = i;
                break;
            }
        }
        if(nodeIndex === -1) return;
        var node = this.mainCanvas.elements[nodeIndex];
        if(node.id === this.startingNode.id) return;
        var isRepeater = (node instanceof RepeaterNode);

        var prevNode = node.getPreviousElement(isRepeater);
        var nextNode = node.getNextElement(isRepeater);
        if(prevNode !== undefined && prevNode !== null) {
            // tail piece; unhook previous line
            prevNode.unhookConnectingLine(false, true, isRepeater);
        }
        if(nextNode !== undefined && nextNode !== null) {
            // starting piece; unhook next line
            nextNode.unhookConnectingLine(true, true, isRepeater);
        }

        // reconnect (if it is not a repeater)
        if(!isRepeater) {
            if(prevNode !== undefined && prevNode !== null && nextNode !== undefined && nextNode !== null) {
                // reconnect previous and next nodes
                var line = new ConnectionLine(this.newID(), this, prevNode, nextNode, false);
                this.bottomCanvas.addElement(line);
                nextNode.setConnectingLine(line, true, false);

            } else if(nextNode !== undefined && nextNode !== null && (prevNode === undefined || prevNode === null)) {
                // reconnect next to latest node in chain
                var lastNode = this._get_last_node();
                var line = new ConnectionLine(this.newID(), this, lastNode, nextNode, false);
                this.bottomCanvas.addElement(line);
                nextNode.setConnectingLine(line, true, false);
            }
        }

        // remove this node
        this.mainCanvas.removeElement(this.mainCanvas.elements[nodeIndex].id);
    }

    removeSelectedNodes() {
        for(var key in this.activeElements) {
            if(key === this.startingNode.id) continue;
            this.removeNode(key);
            this.mainCanvas.removeElement(key);
            this.bottomCanvas.removeElement(key);
        }
    }

    clear() {
        this.mainCanvas.clear(this.startingNode);
        this.bottomCanvas.clear(this.tempConnectionLine);
    }

    toJSON() {
        // put nodes in order first
        var nodeSpec = [];
        var currentNode = this.startingNode;
        do {
            currentNode = currentNode.getNextElement(false);
            if(currentNode !== undefined && currentNode !== null) {
                nodeSpec.push(currentNode.toJSON());
            } else {
                break;
            }
        } while(currentNode !== undefined && currentNode !== null);

        // add repeater nodes
        var repeaterSpec = {};
        for(var n=0; n<this.mainCanvas.elements.length; n++) {
            var node = this.mainCanvas.elements[n];
            if(node instanceof RepeaterNode) {
                var spec = node.toJSON();
                if(spec !== undefined && spec !== null) {
                    repeaterSpec[node.id] = spec;
                }
            }
        }

        // return
        return {
            tasks: nodeSpec,
            repeaters: repeaterSpec
        }
    }
}