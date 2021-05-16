/**
 * 2021 Benjamin Kellenberger
 */

function _apply_columns(entityID, widths, minWidth, ellipsis) {
    let styleTag = $('<style></style>');

    for(var w=0; w<widths.length; w++) {
        let styleStr = entityID + ':nth-child('+(w+1)+') {';
        if(widths[w] !== null) {
            styleStr += 'width: ' + widths[w] + ';';
            if(minWidth) {
                styleStr += 'min-width: ' + widths[w] + ';';
            }
        }
        if(ellipsis) {
            if(widths[w] !== null) {
                styleStr += 'max-width: ' + widths[w] + ';';
            }
            styleStr += 'overflow: hidden;' +
                        'white-space: nowrap;' +
                        'text-overflow: ellipsis;';
        }
        
        styleStr += '}';
        styleTag.append(styleStr);
    }
    $('html > head').append(styleTag);
}

window.adjustTableWidth = function(tableID, widths, minWidth, ellipsis, skipCheck) {
    if(!Array.isArray(widths)) return;

    ellipsis = window.parseBoolean(ellipsis);
    skipCheck = window.parseBoolean(skipCheck);

    if(skipCheck) {
        // assume full table
        _apply_columns(tableID + ' thead tr th', widths, minWidth, ellipsis);
        _apply_columns(tableID + ' tbody tr td', widths, minWidth, ellipsis);
        return;
    }

    // check node type of tableID
    let table = $(tableID);
    let nodeType = table.prop('nodeName').toLowerCase();
    if(nodeType === 'table') {
        // check for separate head and body
        _apply_columns(tableID + ' thead tr th', widths, minWidth, ellipsis);
        _apply_columns(tableID + ' tbody tr td', widths, minWidth, ellipsis);

    } else if(nodeType === 'thead') {
        _apply_columns(tableID + 'thead tr th', widths, minWidth, ellipsis);
    } else if(nodeType === 'tbody') {
        _apply_columns(tableID + 'tbody tr td', widths, minWidth, ellipsis);
    }
}