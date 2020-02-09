/*
    JS module for displaying a given list of images,
    either in list view, or with thumbnails.

    2020 Benjamin Kellenberger
*/


class Thumbnail {
    constructor(imageURI, showCheckbox) {
        this.imageURI = imageURI;
        this.showCheckbox = showCheckbox;

        // setup markup
        this.markup = $('<div class="image-thumbnail"></div>');
        var imageEntry = $('<img>');
        imageEntry.on('error', function() {
            this.src = '/static/dataAdmin/img/notFound.png';
        });
        imageEntry.attr('src', imageURI);
        this.markup.append(imageEntry);

        if(this.showCheckbox) {
            this.checkbox = $('<input type="checkbox" />');
            this.markup.append(this.checkbox);
        }
    }

    isChecked() {
        if(this.checkbox !== undefined) {
            return this.checkbox.prop('checked');
        } else {
            return undefined;
        }
    }
}


class ImageBrowser {
    constructor(div, data) {
        this.div = div;
        this.data = data;

        // setup markup
        var self = this;
        var viewStyle = $('<div></div>');
        this.listViewBtn = $('<button class="btn btn-sm btn-secondary><img src="/static/dataAdmin/img/listView.svg" /></button>');
        this.tileViewBtn = $('<button class="btn btn-sm btn-dark><img src="/static/dataAdmin/img/tileView.svg" /></button>');
        this.listViewBtn.click(function() {
            self.setView('list');
        });
        this.tileViewBtn.click(function() {
            self.setView('tile');
        });
        viewStyle.append(this.listViewBtn);
        viewStyle.append(this.tileViewBtn);
        div.append(viewStyle);

        this.viewPane = $('<div></div>');
        div.append(this.viewPane);

        this.listView = $('<div></div>');
        this.tileView = $('<div></div>');

        this.activeView = 'list';
        this.setView('tile');

        this.dirty = false;     // set to true if images change. If true, list/thumbs will be recreated
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
            this.viewPane.append(this.listView);
        } else {
            $(this.listViewBtn).removeClass('btn-secondary');
            $(this.listViewBtn).addClass('btn-dark');
            $(this.tileViewBtn).removeClass('btn-dark');
            $(this.tileViewBtn).addClass('btn-secondary');
            this.viewPane.empty();
            this.viewPane.append(this.tileView);
        }
        this._populate();
    }

    setImages(images) {
        this.data['images'] = images;
        this.dirty = true;
        this._populate();
    }

    _populate_list() {
        //TODO: thead? Construct table if not existant
        var thead = $(this.listView).find('thead')[0];
        thead.empty();
        var tbody = $(this.listView).find('tbody')[0];
        tbody.empty();
        

    }

    _populate_tiles() {
        //TODO
    }

    _populate() {
        if(!this.dirty) return;
        if(this.activeView === 'list') {
            this._populate_list();
        } else {
            this._populate_tiles();
        }
        this.dirty = false;
    }
}