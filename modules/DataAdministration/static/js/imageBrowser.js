/*
    JS module for displaying a given list of images,
    either in list view, or with thumbnails.

    2020-22 Benjamin Kellenberger
*/

var randomUID = function() {
    return new Date().toString() + Math.random().toString(36).substring(7);
}


class IBProgressBar {
    constructor(visible, value, max) {
        this.value = value;
        this.max = max;

        this.markup = $('<div class="progressbar"></div>');
        this.pbarInd = $('<div class="progressbar-filler progressbar-active"></div>');
        this.markup.append(this.pbarInd);
        if(visible !== undefined) {
            this.markup.css('visibility', (visible? 'visible' : 'hidden'));
        }
    }

    set(visible, value, max) {
        if(visible !== undefined) {
            this.markup.css('visibility', (visible? 'visible' : 'hidden'));
            this.markup.parent().css('visibility', (visible? 'visible' : 'hidden'));
        }
        if(max !== undefined) {
            this.max = max;
        }
        if(value !== undefined) {
            this.value = value;
        }
        var newWidthPerc = (100*(this.value / this.max));
        this.pbarInd.animate({
            'width': newWidthPerc + '%'
        }, 1000);
    }

    getMarkup() {
        return this.markup;
    }
}


class ImageEntry {
    constructor(data, parent, baseURL, showImage, showCheckbox) {
        this.data = data;
        if(!this.data.hasOwnProperty('id')) {
            this.id = randomUID();
        } else {
            this.id = this.data['id'];
        }
        this.parent = parent;
        this.baseURL = baseURL;
        if(this.baseURL === undefined || this.baseURL === null) {
            this.baseURL = '';
        }
        this.showImage = showImage;
        this.showCheckbox = showCheckbox;
        this.selected = false;

        if(this.data.hasOwnProperty('imageURL')) {
            this.imageURL = this.data['imageURL'];
        } else {
            this.imageURL = this.data['url'];
        }

        this.checkbox = undefined;
        this.image = undefined;
        this.markups = {};
    }

    _create_checkbox() {
        if(this.checkbox === undefined) {
            var self = this;
            this.checkbox = $('<input type="checkbox" />');
            this.checkbox.on('change', function() {
                self.parent._on_entry_check($(this).is(':checked'), self);
            });
        }
        return this.checkbox;
    }

    _create_progressBar() {
        if(this.progressBar === undefined) {
            this.progressBar = new IBProgressBar(false, 0, 100);
        }
        return this.progressBar.getMarkup();
    }

    _load_image_contents() {
        if(this.imageLoaded || this.image === undefined) return;

        this.image.attr('src', '/static/dataAdmin/img/loading.png');

        // get ImageRenderer to create canvas from source
        let fullImagePath = this.baseURL+this.imageURL;
        if(typeof(getParserByExtension) === 'function') {
            let parserClass = getParserByExtension(this.imageURL.substring(this.imageURL.lastIndexOf('.')));
            if(parserClass !== undefined) {
                let parser = new parserClass(fullImagePath);
                let self = this;
                parser.get_image_array([0,1,2], false).then((arr) => {          //TODO: get bands from render config
                    // create canvas to put image data to
                    let canvas = document.createElement('canvas');
                    canvas.width = parser.getWidth();
                    canvas.height = parser.getHeight();
                    let imageData = new ImageData(new Uint8ClampedArray(arr), canvas.width, canvas.height);
                    canvas.getContext('2d').putImageData(imageData, 0, 0);
                    self.image.attr('src', canvas.toDataURL());
                    self.imageLoaded = true;
                });
            } else {
                // no parser found; fallback to Web image
                //TODO: test to load image only on click
                this.image.attr('src', fullImagePath);
                this.imageLoaded = true;
            }

        } else {
            // no ImageRenderer available; fallback to Web image
            //TODO: test to load image only on click
            this.image.attr('src', fullImagePath);
            this.imageLoaded = true;
        }
        // this.image.on('click', function(event) {
        //     self.parent._on_entry_click(event, self);
        // });
    }

    _create_image() {
        if(this.image === undefined) {
            let self = this;
            this.image = $('<img src="/static/dataAdmin/img/clicktoload.png" />');
            this.image.on('error', function() {
                if(this.src !== '/static/dataAdmin/img/error.png')
                    this.src = '/static/dataAdmin/img/error.png';
            });

            this.image.on('click', function(event) {
                self._load_image_contents();
                self.parent._on_entry_click(event, self);
            });
        }
        return this.image;
    }


    getMarkup(view) {
        var self = this;
        if(view instanceof ListView) {
            var markup = $('<tr></tr>');
            if(this.showCheckbox) {
                var checkbox = this._create_checkbox();
                var td = $('<td></td>');
                td.append(checkbox);
                markup.append(td);
            }
            if(this.showImage) {
                var image = this._create_image();
                image.addClass('list-entry-thumb');
                var td = $('<td></td>');
                td.append(image);
                markup.append(td);
            }

            for(var j=0; j<view.varOrder.length; j++) {
                var value = this.data[view.varOrder[j]];
                if(value === undefined || value === null) {
                    value = '';
                }
                var td = $('<td class="' + view.varOrder[j] + '">' + value + '</td>');
                td.on('click', function(event) {
                    self.parent._on_entry_click(event, self);
                });
                markup.append(td);
            }
            
            var pBar = this._create_progressBar();
            pBar.addClass('progress-bar-list');
            var td = $('<td style="display:none"></td>');
            td.append(pBar);
            markup.append(td);
            this.markups['list'] = markup;

        } else if(view instanceof ThumbnailView) {
            var markup = $('<div class="thumbnail"></div>');

            var image = this._create_image(view);
            markup.append(image);

            var infoBar = $('<div class="info-bar"></div>');
            if(this.showCheckbox) {
                var checkbox = this._create_checkbox(view);
                infoBar.append(checkbox);
            }
            var span = $('<span class="file-name">'+this.data['url']+'</span>');
            span.on('click', function(event) {
                self.parent._on_entry_click(event, self);
            });
            infoBar.append(span);
            markup.append(infoBar);

            var pBar = this._create_progressBar();
            pBar.addClass('progress-bar-thumbnail');
            markup.append(pBar);

            this.markups['thumbs'] = markup;
        }

        return markup;
    }


    isChecked() {
        if(this.checkbox === undefined) return undefined;
        else return this.checkbox.is(':checked');
    }

    setChecked(checked) {
        if(this.checkbox === undefined) return;
        else this.checkbox.prop('checked', checked);
    }

    isSelected() {
        return this.selected;
    }

    setSelected(selected) {
        this.selected = selected;
        if(this.data['selectable']) {
            if(selected) {
                if('list' in this.markups) {
                    this.markups['list'].addClass('list-entry-selected');
                }
                if('thumbs' in this.markups) {
                    this.markups['thumbs'].addClass('thumbnail-selected');
                }
            } else {
                if('list' in this.markups) {
                    this.markups['list'].removeClass('list-entry-selected');
                }
                if('thumbs' in this.markups) {
                    this.markups['thumbs'].removeClass('thumbnail-selected');
                }
            }
        }
    }

    setProgressBar(visible, value, max) {
        this.progressBar.set(visible, value, max);
    }

    setImageURL(url) {
        this.imageURL = url;
        // this.image.attr('src', this.imageURL);       // on click only

    }

    set(varName, value) {
        if(this.hasOwnProperty(varName)) {
            this[varName] = value;
        }
        if(this.data.hasOwnProperty(varName)) {
            this.data[varName] = value;
        }
        if(varName === 'imageURL') {
            this.setImageURL(value);
        }

        // also modify list markup
        var markupItem = $(this.markups['list']).find('.'+varName);
        if(markupItem.length) {
            $(markupItem[0]).html(value);
        }
    }
}



class AbstractImageView {
    /*
        Abstract base class
    */
    constructor(parent, div, data) {
        this.parent = parent;
        this.div = div;
        this.data = data;
        this.entries = {};
        this.selected = {};
        this.checked = {};

        this.order = {};
        this.order_inv = {};

        this.trailingButton = undefined;
        this.loadingOverlay = undefined;
    }

    containsEntry(key) {
        return this.entries.hasOwnProperty(key);
    }

    addEntries(entries) {
        this.entries = {...this.entries, ...entries};
    }

    addImages(images) {
        var entries = {};
        for(var key in images) {
            var entry = new ImageEntry(images[key],
                                        this,
                                        this.data['baseURL'],
                                        this.data['showImages'],
                                        this.data['showCheckboxes']);
            entries[entry.id] = entry;
        }
        this.addEntries(entries);
    }

    removeEntries(keys) {
        if(typeof(keys) === 'string') keys = [keys];
        for(var k in keys) {
            if(this.containsEntry(keys[k])) {
                delete this.entries[keys[k]];
            }
        }
    }

    getEntry(idx) {
        if(this.order_inv.hasOwnProperty(idx)) {
            return this.entries[this.order_inv[idx]];
        } else {
            return null;
        }
    }

    setEntries(entries) {
        this.entries = entries;
    }

    setImages(images) {
        for(var key in this.entries) {
            delete this.entries[key];
        }
        this.entries = {};
        this.addImages(images);
    }

    getSelected() {
        var selected = [];
        for(var key in this.entries) {
            if(this.entries[key].isSelected()) {
                selected.push(this.entries[key]);
            }
        }
        return selected;
    }

    getChecked() {
        var checked = [];
        for(var key in this.entries) {
            if(this.entries[key].isChecked()) {
                checked.push(this.entries[key]);
            }
        }
        return checked;
    }

    setTrailingButton(visible, disabled, buttonText, callback) {
        error('Not implemented for abstract base class.');
    }

    setProgressBar(entryID, visible, value, min, max) {
        if(this.entries.hasOwnProperty(entryID)) {
            this.entries[entryID].setProgressBar(visible, value, min, max);
        }
    }

    setLoadingOverlay(visible, text, value, max) {
        /**
         * Either displays loading dots (if no value, min and max are provided)
         * or shows a progress bar with given parameters.
         */
        let showPbar = (typeof(value) === 'number');
        if(this.loadingOverlay === undefined) {
            this.loadingOverlay = $('<div class="loading-overlay"></div>');
            let loDecoration = $('<div class="loading-overlay-decoration"></div>');
            this.loadingPbar = new IBProgressBar(showPbar, value, max);
            loDecoration.append(this.loadingPbar.getMarkup());
            this.loadingDots = $('<div class="azure-loadingdots">' +
                                        '<div></div>' +
                                        '<div></div>' +
                                        '<div></div>' +
                                        '<div></div>' +
                                        '<div></div>' +
                                        '</div>');
            this.loadingDots.css('visibility', (showPbar? 'hidden' : 'visible'));
            loDecoration.append(this.loadingDots);
            this.loadingOverlayText = $('<div class="loading-overlay-text"></div>');
            loDecoration.append(this.loadingOverlayText);
            this.loadingOverlay.append(loDecoration);
            this.div.append(this.loadingOverlay);
        }
        if(visible) {
            if(showPbar) {
                this.loadingDots.css('visibility', 'hidden');
                this.loadingPbar.set(true, value, max);
            } else {
                this.loadingDots.css('visibility', 'visible');
                this.loadingPbar.set(false);
            }
        } else if(visible !== undefined) {
            this.loadingDots.css('visibility', 'hidden');
            this.loadingPbar.set(false);
            this.loadingOverlay.css('visibility', (visible? 'visible' : 'hidden'));
        }
        if(text !== undefined) this.loadingOverlayText.html(text === null? '' : text);
    }

    get(varName, entry) {
        if(this.entries.hasOwnProperty(entry) && this.entries[entry].hasOwnProperty(varName)) {
            return this.entries[entry][varName];
        } else {
            return undefined;
        }
    }

    set(varName, entry, value) {
        if(this.entries.hasOwnProperty(entry)) {
            if(varName === 'imageURL') {
                this.entries[entry].setImageURL(value);
            } else {
                this.entries[entry].set(varName, value);
            }
        }
    }

    _on_entry_click(event, entry) {
        error('Not implemented for abstract base class.');
    }

    _on_entry_check(checked, entry) {
        var affected = [entry];

        // also check (or uncheck) selected entries
        for(var key in this.selected) {
            this.selected[key].setChecked(checked);
            if(checked) {
                this.checked[key] = this.selected[key];
            } else {
                delete this.checked[key];
            }
            affected.push(this.selected[key]);
        }
        if(checked) {
            this.checked[entry.id] = entry;
        } else {
            delete this.checked[entry.id];
        }

        // fire event
        this.parent._fire_event('imageCheck', affected);
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

    _fire_event(event, object) {
        this.parent._fire_event(event, object);
    }
}


class ThumbnailView extends AbstractImageView {
    constructor(parent, div, data) {
        super(parent, div, data);
        this._setup_markup();
    }

    addEntries(entries) {
        super.addEntries(entries);
        if(this.trailingButton !== undefined && $.contains(document, this.trailingButton[0])) this.trailingButton.detach();
        var idx = Object.keys(this.order).length;
        for(var key in entries) {
            var markup = this.entries[key].getMarkup(this);
            this.thumbsContainer.append(markup);
            this.order[this.entries[key].id] = idx;
            this.order_inv[idx] = this.entries[key].id;
            idx++;
        }
        if(this.trailingButton !== undefined) this.thumbsContainer.append(this.trailingButton);
    }

    setEntries(entries) {
        if(this.trailingButton !== undefined) this.trailingButton.detach();
        this.thumbsContainer.empty();
        super.setEntries(entries);
        this._setup_markup();
        if(this.trailingButton !== undefined) this.thumbsContainer.append(this.trailingButton);
    }

    setImages(images) {
        if(this.trailingButton !== undefined) this.trailingButton.detach();
        this.thumbsContainer.empty();
        super.setImages(images);
        if(this.trailingButton !== undefined) this.thumbsContainer.append(this.trailingButton);
    }

    setTrailingButton(visible, disabled, buttonText, callback) {
        if(this.trailingButton === undefined) {
            // create button
            this.trailingButton = $('<button class="btn btn-secondary thumbnail"></button>');
            this.thumbsContainer.append(this.trailingButton);
        }
        if(visible !== undefined) this.trailingButton.css('visibility', (visible? 'visible' : 'hidden'));
        if(disabled !== undefined) this.trailingButton.prop('disabled', (disabled? true : false));
        if(buttonText !== undefined) this.trailingButton.html(buttonText === null? '' : buttonText);
        if(callback !== undefined) this.trailingButton.on('click', function(event) {
            callback(event);
        });
    }

    _setup_markup() {
        if(this.thumbsContainer === undefined) {
            this.thumbsContainer = $('<div class="image-browser-thumbs-container"></div>');
            this.div.append(this.thumbsContainer);

            // image size slider
            let iszrange = $('<input type="range" min="24" max="1024" value="128" />');
            iszrange.on('input', function() {
                let height = $(this).val();
                let ratio = $('.thumbnail').width() / $('.thumbnail').height();
                let width = height * ratio;
                $('.thumbnail').css('height', height.toString()+'px');
                $('.thumbnail').css('width', width.toString()+'px');
            });
            let iszrangeCont = $('<div class="image-browser-size-panel">image size:</div>');
            iszrangeCont.append(iszrange);
            this.div.append(iszrangeCont);
        }
        this.thumbsContainer.empty();
        this.order = {};
        this.order_inv = {};
        var idx = 0;
        for(var key in this.entries) {
            var markup = this.entries[key].getMarkup(this);
            this.thumbsContainer.append(markup);
            this.order[this.entries[key].id] = idx;
            this.order_inv[idx] = this.entries[key].id;
            idx++;
        }
    }

    _on_entry_click(event, entry) {
        var selID = entry.id;
        var wasSelected = (selID in this.selected);
        var affected = [];
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
                var isSel = this.entries[this.order_inv[i]].isSelected();
                if(isSel) {
                    this.entries[this.order_inv[i]].setSelected(false);
                    delete this.selected[this.order_inv[i]];
                } else {
                    this.entries[this.order_inv[i]].setSelected(true);
                    this.selected[this.order_inv[i]] = this.entries[this.order_inv[i]];
                }
                affected.push(this.entries[this.order_inv[i]]);
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
            entry.setSelected(!wasSelected);
            if(!wasSelected) {
                this.selected[selID] = entry;
            } else {
                delete this.selected[selID];
            }
            affected.push(entry);
        }

        this._fire_event('imageClick', affected);
    }
}


class ListView extends AbstractImageView {
    constructor(parent, div, data) {
        super(parent, div, data);
        this._setup_markup();
    }

    addEntries(entries) {
        super.addEntries(entries);
        if(this.trailingButton !== undefined && $.contains(document, this.trailingButton[0])) this.trailingButton.detach();
        var idx = Object.keys(this.order).length;
        for(var key in entries) {
            var markup = this.entries[key].getMarkup(this);
            this.tbody.append(markup);
            this.order[this.entries[key].id] = idx;
            this.order_inv[idx] = this.entries[key].id;
            idx++;
        }
        if(this.trailingButton !== undefined) this.tbody.append(this.trailingButton);
    }

    setEntries(entries) {
        if(this.trailingButton !== undefined) this.trailingButton.detach();
        this.tbody.empty();
        super.setEntries(entries);
        this._setup_markup();
        if(this.trailingButton !== undefined) this.tbody.append(this.trailingButton);
    }

    setImages(images) {
        if(this.trailingButton !== undefined) this.trailingButton.detach();
        this.tbody.empty();
        super.setImages(images);
        if(this.trailingButton !== undefined) this.tbody.append(this.trailingButton);
    }

    setTrailingButton(visible, disabled, buttonText, callback) {
        if(this.trailingButton === undefined) {
            // create button
            this.trailingButton = $('<button class="btn btn-secondary list-trailing-button"></button>');
            this.tbody.append(this.trailingButton);
        }
        if(visible !== undefined) this.trailingButton.css('visibility', (visible? 'visible' : 'hidden'));
        if(disabled !== undefined) this.trailingButton.prop('disabled', (disabled? true : false));
        if(buttonText !== undefined) this.trailingButton.html(buttonText === null? '' : buttonText);
        if(callback !== undefined) this.trailingButton.on('click', function(event) {
            callback(event);
        });
    }

    _setup_markup() {
        if(this.tbody === undefined) {
            // assemble table
            var self = this;
            this.varOrder = [];
            var thead = $('<thead class="list-header"></thead>');
            var tr = $('<tr></tr>');
            thead.append(tr);

            if(this.data['showCheckboxes']) {
                this.checkAll = $('<input type="checkbox" />');
                this.checkAll.on('click', function() {
                    self._check_all();
                });
                var cell = $('<th></th>');
                cell.append(this.checkAll);
                tr.append(cell);
            }
            if(this.data['showImages']) {
                tr.append($('<th></th>'));
            }
            for(var i=0; i<this.data['colnames'].length; i++) {
                var nextCol = this.data['colnames'][i];
                var nextKey = Object.keys(nextCol)[0];
                this.varOrder.push(nextKey);
                tr.append($('<th>' + nextCol[nextKey] + '</th>'));
            }
            // for progress bar
            tr.append($('<th></th>'));
            var table = $('<table class="list-table"></table>');
            table.append(thead);
            this.tbody = $('<tbody class="list-body"></tbody>');
            table.append(this.tbody);
            this.div.append(table);

            // image size slider
            let iszrange = $('<input type="range" min="24" max="256" value="50" />');
            iszrange.on('input', function() {
                let height = $(this).val();
                $('.list-entry-thumb').css('height', height.toString()+'px');
            });
            let iszrangeCont = $('<div class="image-browser-size-panel">image size:</div>');
            iszrangeCont.append(iszrange);
            this.div.append(iszrangeCont);
        }

        this.tbody.empty();
        this.order = {};
        this.order_inv = {};
        var idx = 0;
        for(var key in this.entries) {
            var markup = this.entries[key].getMarkup(this);
            this.tbody.append(markup);
            this.order[this.entries[key].id] = idx;
            this.order_inv[idx] = this.entries[key].id;
            idx++;
        }
    }
    
    _check_all() {
        var checked = this.checkAll.prop('checked');
        var affected = [];
        for(var key in this.entries) {
            this.entries[key].setChecked(checked);
            affected.push(this.entries[key]);
        }
        this.parent._fire_event('imageCheck', affected);
    }

    _on_entry_click(event, entry) {
        var selID = entry.id;
        var wasSelected = (selID in this.selected);
        var affected = [];
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
                var isSel = this.entries[this.order_inv[i]].isSelected();
                if(isSel) {
                    this.entries[this.order_inv[i]].setSelected(false);
                    delete this.selected[this.order_inv[i]];
                } else {
                    this.entries[this.order_inv[i]].setSelected(true);
                    this.selected[this.order_inv[i]] = this.entries[this.order_inv[i]];
                }
                affected.push(this.entries[this.order_inv[i]]);
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
            entry.setSelected(!wasSelected);
            if(!wasSelected) {
                this.selected[selID] = entry;
            } else {
                delete this.selected[selID];
            }
            affected.push(entry);
        }

        this._fire_event('imageClick', affected);
    }
}



class ImageBrowser {
    /*
        Combines the different views into a single panel.
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
        if(!this.data.hasOwnProperty('selectable')) {
            this.data['selectable'] = true;
        }

        this.callbacks = {
            viewChange: [],
            imageUpdate: [],
            imageClick: [],
            imageCheck: []
        };

        // setup markup
        var self = this;
        var viewStyle = $('<div class="image-browser-view-buttons"></div>');
        this.listViewBtn = $('<button class="btn btn-sm btn-secondary" id="list-view-button"><img src="/static/dataAdmin/img/listView.svg" height="12" /></button>');
        this.tileViewBtn = $('<button class="btn btn-sm btn-dark" id="tile-view-button"><img src="/static/dataAdmin/img/tileView.svg" height="12" /></button>');
        this.listViewBtn.click(function() {
            self.setView('list');
        });
        this.tileViewBtn.click(function() {
            self.setView('tile');
        });
        viewStyle.append(this.listViewBtn);
        viewStyle.append(this.tileViewBtn);
        div.append(viewStyle);

        this.viewPane = $('<div style="border:1px solid #aaa;"></div>');
        div.append(this.viewPane);


        this.listViewDiv = $('<div class="list-container"></div>');
        this.tileViewDiv = $('<div class="thumbs-container"></div>');

        this.listView = new ListView(this, this.listViewDiv, this.data);
        this.tileView = new ThumbnailView(this, this.tileViewDiv, this.data);

        
        if(this.data.hasOwnProperty('images')) {
            this.setImages(this.data['images']);
        }


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
            this.tileViewDiv.detach();
            this.viewPane.append(this.listViewDiv);
        } else {
            $(this.listViewBtn).removeClass('btn-secondary');
            $(this.listViewBtn).addClass('btn-dark');
            $(this.tileViewBtn).removeClass('btn-dark');
            $(this.tileViewBtn).addClass('btn-secondary');
            this.listViewDiv.detach();
            this.viewPane.append(this.tileViewDiv);
        }

        this._fire_event('viewChange', type)
    }

    containsEntry(key) {
        if(this.activeView === 'list') {
            return this.listView.containsEntry(key);
        } else {
            return this.tileView.containsEntry(key);
        }
    }

    addImages(images) {
        this.data['images'] = {...this.data['images'], ...images};
        this.listView.addImages(images);
        this.tileView.addImages(images);
    }

    setImages(images) {
        this.data['images'] = images;
        this.listView.setImages(images);
        this.tileView.setImages(images);
        if(images === undefined || images === null || images.length === 0) {
            this.setTrailingButton(false);
        }
    }

    removeEntries(keys) {
        if(this.activeView === 'list') {
            return this.listView.removeEntries(keys);
        } else {
            return this.tileView.removeEntries(keys);
        }
    }

    getSelected() {
        if(this.activeView === 'list') {
            return this.listView.getSelected();
        } else {
            return this.tileView.getSelected();
        }
    }

    getChecked() {
        if(this.activeView === 'list') {
            return this.listView.getChecked();
        } else {
            return this.tileView.getChecked();
        }
    }

    getEntries() {
        return this.listView.entries;
    }

    getNumEntries() {
        return Object.keys(this.listView.entries).length;
    }

    getEntry(idx) {
        return this.listView.getEntry(idx);
    }

    get(varName, entry) {
        return this.listView.get(varName, entry);
    }

    set(varName, entry, value) {
        this.listView.set(varName, entry, value);
        this.tileView.set(varName, entry, value);
    }

    setTrailingButton(visible, disabled, buttonText, callback) {
        /*
            Adds a button at the end of the list (or thumbnails)
            that can be customized.
        */
        this.listView.setTrailingButton(visible, disabled, buttonText, callback);
        this.tileView.setTrailingButton(visible, disabled, buttonText, callback);
    }

    setLoadingOverlay(visible, text, value, max) {
        /*
            Semi-transparent overlay over the image browser views.
        */
        this.listView.setLoadingOverlay(visible, text, value, max);
        this.tileView.setLoadingOverlay(visible, text, value, max);
    }

    setProgressBar(entryID, visible, value, max) {
        this.listView.setProgressBar(entryID, visible, value, max);
        this.tileView.setProgressBar(entryID, visible, value, max);
    }

    // event handling
    _on_entry_check(checked, entry) {
        if(this.activeView === 'list') {
            this.listView._on_entry_check(checked, entry);
        } else {
            this.tileView._on_entry_check(checked, entry);
        }
    }

    _on_entry_click(event, entry) {
        if(this.activeView === 'list') {
            this.listView._on_entry_click(event, entry);
        } else {
            this.tileView._on_entry_click(event, entry);
        }
    }

    on(eventType, functionHandle) {
        if(this.callbacks.hasOwnProperty(eventType)) {
            this.callbacks[eventType].push(functionHandle);
        }
    }

    _fire_event(eventType, object) {
        if(this.callbacks.hasOwnProperty(eventType)) {
            for(var i=0; i<this.callbacks[eventType].length; i++) {
                this.callbacks[eventType][i](object);
            }
        }
    }
}