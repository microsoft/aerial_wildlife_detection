/**
 * Messaging system for the user front-end.
 * 
 * 2020-21 Benjamin Kellenberger
 */


class Message {

    constructor(text, type, fadeOutTime) {
        this.text = text;
        this.type = (['success', 'warning', 'error', 'info'].includes(type) ? type : 'regular');
        this.fadeOutTime = (typeof(fadeOutTime) === 'number' ? fadeOutTime : 10000);

        // setup markup
        this.markup = $('<div class="messager-message messager-message-'+this.type+'"></div>');
        
        let body = $('<div class="messager-message-body"></div>');
        this.markup.append(body);

        let textDiv = $('<div class="messager-message-text">'+this.text+'</div>');
        body.append(textDiv);

        let closeButton = $('<button class="messager-message-close-button btn btn-sm btn-secondary">X</button>');
        closeButton.on('click', (function() {
            this.setVisible(false);
        }).bind(this));
        body.append(closeButton);

        this.setVisible(true);
    }

    setVisible(visible) {
        if(visible) {
            if(this.fadeOutTime > 0) {
                this.timeoutHandle = setTimeout((function() {
                    this.setVisible(false);
                }).bind(this), this.fadeOutTime);
            }
            this.markup.slideDown().fadeIn();
        } else {
            this.markup.slideUp().fadeOut();
            clearTimeout(this.timeoutHandle);
        }
    }
}


class Messager {

    constructor(domElement) {
        this.domElement = $(domElement);

        // add filler at the top
        this.domElement.empty();
        this.domElement.append($('<div class="messager-filler"></div>'));

        this.messages = [];
    }

    addMessage(text, type, fadeOutTime) {
        let message = new Message(text, type, fadeOutTime);
        this.messages.push(message);
        this.domElement.append($(message.markup));
        message.setVisible(true);
    }
}


$(document).ready(function() {
    window.messager = new Messager($('#messager-container'));
});