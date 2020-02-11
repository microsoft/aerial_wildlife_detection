/*
    JS module for displaying a given list of images,
    either in list view, or with thumbnails.

    2020 Benjamin Kellenberger
*/

var randomUID = function() {
    return new Date().toString() + Math.random().toString(36).substring(7);
}


class AbstractEntry {
    constructor(data, parent, baseURL, showImage, showCheckbox) {
        this.data = data;
        if(!this.data.hasOwnProperty('id')) {
            this.id = randomUID();
        } else {
            this.id = this.data['id'];
        }
        this.parent = parent;
        this.baseURL = baseURL;
        this.showImage = showImage;
        this.showCheckbox = showCheckbox;
        this.checked = false;
        this.selected = false;
    }

    isChecked() {
        //TODO
        // if(this.checkbox !== undefined) {
        //     return this.checkbox.prop('checked');
        // } else {
        //     return undefined;
        // }
    }

    setChecked(checked) {
        this.checked = checked;
    }

    isSelected() {
        return this.selected;
    }

    setSelected(selected) {
        this.selected = selected;
    }
}


class ListEntry extends AbstractEntry {
    constructor(data, parent, baseURL, showImage, showCheckbox, varOrder) {
        super(data, parent, baseURL, showImage, showCheckbox);
        this.baseURL = baseURL;
        this.varOrder = varOrder;
        this.showImage = showImage;

        // setup markup
        var self = this;
        this.markup = $('<tr></tr>');
        if(showCheckbox) {
            this.checkbox = $('<input type="checkbox" />');
            this.checkbox.on('change', function() {
                self.parent._on_entry_check(self.checkbox.is(':checked'), self);
            });
            var td = $('<td></td>');
            td.append(this.checkbox);
            this.markup.append(td);
        }
        if(showImage) {
            var imageEntry = $('<img class="list-entry-thumb">');
            imageEntry.on('error', function() {
                this.src = '/static/dataAdmin/img/notFound.png';
            });
            imageEntry.attr('src', this.baseURL+this.data['url']);
            var td = $('<td></td>');
            td.append(imageEntry);
            this.markup.append(td);
        }
        for(var j=0; j<this.varOrder.length; j++) {
            var value = this.data[this.varOrder[j]];
            if(value === undefined || value === null) {
                value = '';
            }
            this.markup.append($('<td>' + value + '</td>'));
        }

        this.markup.on('click', function(event) {       //TODO: also fires on checkbox click...
            self.parent._on_entry_click(event, self);
        });
    }

    setSelected(selected) {
        super.setSelected(selected);
        if(this.selected) {
            this.markup.addClass('list-entry-selected');
        } else {
            this.markup.removeClass('list-entry-selected');
        }
    }

    setChecked(checked) {
        super.setChecked(checked);
        this.checkbox.prop('checked', checked);
    }
}


class Thumbnail extends AbstractEntry {
    constructor(data, parent, baseURL, showCheckbox) {
        super(data, parent, baseURL, true, showCheckbox);
        this.baseURL = baseURL;        

        // setup markup
        var self = this;
        this.markup = $('<div class="thumbnail"></div>');
        var imageEntry = $('<img>');
        imageEntry.on('error', function() {
            this.src = '/static/dataAdmin/img/notFound.png';
        });
        imageEntry.attr('src', this.baseURL+this.data['url']);
        this.markup.append(imageEntry);

        var infoBar = $('<div class="info-bar"></div>');
        if(this.showCheckbox) {
            this.checkbox = $('<input type="checkbox" />');
            this.checkbox.on('change', function() {
                self.parent._on_entry_check(self.checkbox.is(':checked'), self);
            });
            infoBar.append(this.checkbox);
        }
        infoBar.append($('<span class="file-name">'+this.data['url']+'</span>'));
        this.markup.append(infoBar);

        this.selected = false;
        
        imageEntry.on('click', function(event) {
            self.parent._on_entry_click(event, self);
        });
    }

    setSelected(selected) {
        super.setSelected(selected);
        if(this.selected) {
            this.markup.addClass('thumbnail-selected');
        } else {
            this.markup.removeClass('thumbnail-selected');
        }
    }

    setChecked(checked) {
        super.setChecked(checked);
        this.checkbox.prop('checked', checked);
    }
}


class AbstractImageView {
    /*
        Abstract base class
    */
    constructor(div, data) {
        this.div = div;
        this.data = data;
        if(!this.data.hasOwnProperty('baseURL')) {
            this.data['baseURL'] = '';
        }
        if(this.data['baseURL'].length > 0 && !this.data['baseURL'].endsWith('/')) {
            this.data['baseURL'] = this.data['baseURL'] + '/';
        }
        if(!this.data.hasOwnProperty('showCheckboxes')) {
            this.data['showCheckboxes'] = false;
        }
        if(!this.data.hasOwnProperty('showImages')) {
            this.data['showImages'] = false;
        }

        this.selected = {};
        this.checked = {};
    }

    setImages(images) {
        this.data['images'] = images;
        this._populate();
    }

    _populate() {
        error('Not implemented for abstract base class.');
    }

    _on_entry_click(event, entry) {
        error('Not implemented for abstract base class.');
    }

    _on_entry_check(checked, entry) {
        // also check (or uncheck) selected entries
        for(var key in this.selected) {
            this.selected[key].setChecked(checked);
            if(checked) {
                this.checked[key] = this.selected[key];
            } else {
                delete this.checked[key];
            }
        }
        if(checked) {
            this.checked[entry.id] = entry;
        } else {
            delete this.checked[entry.id];
        }
    }

    _clear_selected() {
        for(var key in this.selected) {
            this.selected[key].setSelected(false);
        }
        this.selected = {};
    }

    _clear_checked() {
        for(var key in this.checked) {
            this.checked[key].setChecked(false);
        }
        this.checked = {};
    }
}


class ThumbnailView extends AbstractImageView {
    constructor(div, data) {
        super(div, data);
        this._populate();
    }

    _populate() {
        this.thumbnails = {};
        this.order = {};
        this.order_inv = {};
        this.div.empty();
        if(!this.data.hasOwnProperty('images') || this.data['images'].length===0) return;
        for(var i=0; i<this.data['images'].length; i++) {
            var thumbnail = new Thumbnail(this.data['images'][i], this, this.data['baseURL'], this.data['showCheckboxes']);
            this.thumbnails[thumbnail.id] = thumbnail;
            this.div.append(thumbnail.markup);
            this.order[thumbnail.id] = i;
            this.order_inv[i] = thumbnail.id;
        }
    }

    _on_entry_click(event, thumbnail) {
        var selID = thumbnail.id;
        var wasSelected = (selID in this.selected);
        if(event.shiftKey) {
            // determine positions
            var selIndex = this.order[selID];
            var minIndex = 1e12;
            var maxIndex = 0;
            for(var key in this.selected) {
                if(this.order[key] < minIndex) {
                    minIndex = this.order[key];
                }
                if(this.order[key] > maxIndex) {
                    maxIndex = this.order[key];
                }
            }

            // expand or reduce selection
            if(selIndex < minIndex) {
                var start = selIndex;
                var end = minIndex - 1;
            } else if(selIndex > maxIndex) {
                var start = maxIndex + 1;
                var end = selIndex;
            } else if(Math.abs(selIndex - minIndex) < Math.abs(selIndex - maxIndex)) {
                var start = minIndex + 1;
                var end = selIndex;
            } else {
                var start = selIndex;
                var end = maxIndex - 1;
            }
            for(var i=start; i<=end; i++) {
                var isSel = this.thumbnails[this.order_inv[i]].isSelected();
                if(isSel) {
                    this.thumbnails[this.order_inv[i]].setSelected(false);
                    delete this.selected[this.order_inv[i]];
                } else {
                    this.thumbnails[this.order_inv[i]].setSelected(true);
                    this.selected[this.order_inv[i]] = this.thumbnails[this.order_inv[i]];
                }
            }

        } else {
            if(!event.ctrlKey && !event.metaKey) {
                // clear selected entries first
                for(var key in this.selected) {
                    this.selected[key].setSelected(false);
                }
                this.selected = {};
            }

            // apply
            thumbnail.setSelected(!wasSelected);
            if(!wasSelected) {
                this.selected[selID] = thumbnail;
            } else {
                delete this.selected[selID];
            }
        }
    }
}


class ListView extends AbstractImageView {
    constructor(div, data) {
        super(div, data);

        // assemble table
        var self = this;
        this.varOrder = [];
        var thead = $('<thead class="list-header"></thead>');
        var tr = $('<tr></tr>');
        thead.append(tr);
        if(this.data['showCheckboxes']) {
            var selectAll = $('<input type="checkbox" />');
            selectAll.on('click', function() {
                self._select_all();
            });
            var cell = $('<td></td>');
            cell.append(selectAll);
            tr.append(cell);
        }
        if(this.data['showImages']) {
            tr.append($('<td></td>'));
        }
        for(var i=0; i<this.data['colnames'].length; i++) {
            var nextCol = this.data['colnames'][i];
            var nextKey = Object.keys(nextCol)[0];
            this.varOrder.push(nextKey);
            tr.append($('<td>' + nextCol[nextKey] + '</td>'));
        }
        var table = $('<table class="list-table"></table>');
        table.append(thead);
        this.tbody = $('<tbody class="list-body"></tbody>');
        table.append(this.tbody);
        this.div.append(table);

        this._populate();
    }

    _populate() {
        this.tbody.empty();
        if(!this.data.hasOwnProperty('images') || this.data['images'].length===0) return;
        
        for(var i=0; i<this.data['images'].length; i++) {
            var entry = new ListEntry(this.data['images'][i], this, this.data['baseURL'], this.data['showImages'], this.data['showCheckboxes'], this.varOrder);
            this.tbody.append(entry.markup);
        }
    }

    _select_all() {
        //TODO
    }


    _on_entry_click(event, entry) {
        //TODO
    }
}



class ImageBrowser {
    /*
        Combines the different views into a single panel.
    */
    constructor(div, data) {
        this.div = div;
        this.data = data;

        // setup markup
        var self = this;
        var viewStyle = $('<div class="image-browser-view-buttons"></div>');
        this.listViewBtn = $('<button class="btn btn-sm btn-secondary"><img src="/static/dataAdmin/img/listView.svg" height="12" /></button>');
        this.tileViewBtn = $('<button class="btn btn-sm btn-dark"><img src="/static/dataAdmin/img/tileView.svg" height="12" /></button>');
        this.listViewBtn.click(function() {
            self.setView('list');
        });
        this.tileViewBtn.click(function() {
            self.setView('tile');
        });
        viewStyle.append(this.listViewBtn);
        viewStyle.append(this.tileViewBtn);
        div.append(viewStyle);

        this.viewPane = $('<div style="height:100%;border:1px solid #aaa;"></div>');
        div.append(this.viewPane);


        this.listViewDiv = $('<div class="list-container"></div>');
        this.tileViewDiv = $('<div class="thumbs-container"></div>');

        this.listView = new ListView(this.listViewDiv, data);
        this.tileView = new ThumbnailView(this.tileViewDiv, data);

        this.setView('list');
    }


    setView(type) {
        if(this.activeView === type) {
            return;
        }
        this.activeView = type;
        if(type === 'list') {
            $(this.listViewBtn).removeClass('btn-dark');
            $(this.listViewBtn).addClass('btn-secondary');
            $(this.tileViewBtn).removeClass('btn-secondary');
            $(this.tileViewBtn).addClass('btn-dark');
            this.viewPane.empty();
            this.viewPane.append(this.listViewDiv);
        } else {
            $(this.listViewBtn).removeClass('btn-secondary');
            $(this.listViewBtn).addClass('btn-dark');
            $(this.tileViewBtn).removeClass('btn-dark');
            $(this.tileViewBtn).addClass('btn-secondary');
            this.viewPane.empty();
            this.viewPane.append(this.tileViewDiv);
        }
    }


    getSelected() {
        if(!Object.hasOwnProperty(this.data['showCheckboxes']) || !this.data['showCheckboxes']) {
            return null;
        }

        if(this.activeView === 'list') {
            return this.listView.getSelected();
        } else {
            return this.tileView.getSelected();
        }
    }
}