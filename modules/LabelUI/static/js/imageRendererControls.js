/**
 * Image renderer driver selection, as well as UI functionality to adjust how
 * images are rendered.
 *
 * 2021-23 Benjamin Kellenberger
 */

async function rerenderAll() {
    /**
     * Calls all image renderers to re-render (e.g., if render config values
     * changed).
     */
    for(var e in window.dataHandler.dataEntries) {
        window.dataHandler.dataEntries[e].renderer.rerenderImage();
    }
}


function setupImageRendererControls(parentDiv, showApplyButtons) {

    // get image format for project and select renderer appropriately.
    window.bandConfig = get_band_config([]);
    window.renderConfig = get_render_config(window.bandConfig, {});
    window.renderConfig_default = get_render_config(window.bandConfig, {});
    return $.ajax({
        url: window.baseURL + 'getConfig',
        method: 'POST',
        data: JSON.stringify({
            'parameters': [
                'band_config',
                'render_config'
            ]
        }),
        contentType: 'application/json; charset=utf-8',
        dataType: 'json',
        success: function(data) {
            try {
                window.bandConfig = get_band_config(data['settings']['band_config']);

                window.renderConfig = data['settings']['render_config'];
                if(typeof(window.renderConfig) !== 'object' || window.renderConfig === null) {
                    window.renderConfig = get_render_config(window.bandConfig, {});
                } else {
                    window.renderConfig = get_render_config(window.bandConfig, window.renderConfig);
                }
                window.renderConfig_default = JSON.parse(JSON.stringify(window.renderConfig));

                // create UI controls w.r.t. capabilities of renderer
                let renderCapabilities = ImageRenderer.prototype.get_render_capabilities();
                let markup = $('<div class="image-renderer-controls"></div>');
                if(renderCapabilities['bands']) {
                    // band configuration
                    let bandConfCont = $('<table></table>');
                    ['Red', 'Green', 'Blue'].map((band) => {
                        let selID = 'band-select-'+band;
                        let select = $('<select id="'+selID.toLowerCase()+'"></select>');
                        for(var l=0; l<window.bandConfig.length; l++) {
                            let label = window.bandConfig[l];
                            let option = $('<option value="'+l+'">'+(l+1) + ': ' + label+'</option>');
                            select.append(option);
                        }
                        let selBand = renderConfig['bands']['indices'][band.toLowerCase()];
                        select.val(selBand.toString());
                        select.on('change', function() {
                            let bandID = $(this).attr('id').replace('band-select-','');
                            window.renderConfig['bands']['indices'][bandID] = parseInt(this.value);
                            // rerenderAll();
                        });
                        let selTd = $('<td></td>');
                        selTd.append(select);
                        let labelTd = $('<td><label for="'+selID+'">'+band+':</label></td>');
                        let selTr = $('<tr></tr>');
                        selTr.append(labelTd);
                        selTr.append(selTd);
                        bandConfCont.append(selTr);
                    });
                    markup.append($('<h3>Bands</h3>'));
                    markup.append(bandConfCont);
                }
                if(renderCapabilities['contrast']) {
                    // contrast stretch
                    let contrastCont = $('<div></div>');
                    contrastCont.append($('<h3>Contrast stretch</h3>'));
                    //TODO
                    if(renderCapabilities['contrast']['percentile']) {
                        //TODO
                        let percTable = $('<table></table>');
                        ['Min', 'Max'].map((valID) => {
                            let row = $('<tr></tr>');
                            row.append($('<td>'+valID+':</td>'));
                            let val = get_render_config_val(window.renderConfig, ['contrast', 'percentile', valID.toLowerCase()], 2.0);
                            let sel = $('<input id="contrast-stretch-perc-'+valID.toLowerCase()+'" type="number" min="0" max="100" value="' + val + '" />');
                            sel.on('focusout', function() {
                                let minSel = $('#contrast-stretch-perc-min');
                                let maxSel = $('#contrast-stretch-perc-max');
                                if($(this).attr('id') === 'contrast-stretch-perc-min') {
                                    $(this).val(Math.min(parseInt($(this).val()), parseInt(maxSel.val())-1));
                                    window.renderConfig['contrast']['percentile']['min'] = parseInt($(this).val());
                                } else {
                                    $(this).val(Math.max(parseInt($(this).val()), parseInt(minSel.val())+1));
                                    window.renderConfig['contrast']['percentile']['max'] = parseInt($(this).val());
                                }
                                // rerenderAll();
                            });
                            let selTd = $('<td></td>');
                            selTd.append(sel);
                            row.append(selTd);
                            percTable.append(row);
                        });
                        contrastCont.append(percTable);
                    }
                    if(contrastCont.children().length) {
                        markup.append(contrastCont);
                    }
                }
                if(renderCapabilities['brightness']) {
                    let brCont = $('<div></div>');
                    brCont.append($('<label for="render-brightness-slider">Brightness:</label>'));
                    let brSlider = $('<input id="render-brightness-slider" type="range" min="0" max="255" value="128" />');
                    let brNum = $('<span style="margin-left:10px">0%</span>');
                    brSlider.on('input', function() {
                        let val = $(this).val() - 128;
                        let val_perc = roundNumber(100*val/255.0, 100);
                        brNum.html(val_perc + '%');
                        window.renderConfig['brightness'] = val;
                    });
                    brCont.append(brSlider);
                    brCont.append(brNum);
                    markup.append(brCont);
                }
                if(renderCapabilities['grayscale']) {
                    let chckbxCont = $('<div></div>');
                    let gsChck = $('<input type="checkbox" id="render-grayscale-checkbox" />');
                    gsChck.on('change', function() {
                        window.renderConfig['grayscale'] = $(this).prop('checked');
                    });
                    chckbxCont.append(gsChck);
                    chckbxCont.append($('<label for="render-grayscale-checkbox">grayscale</label>'));
                    markup.append(chckbxCont);
                }
                if(renderCapabilities['white_on_black']) {
                    let chckbxCont = $('<div></div>');
                    let gsChck = $('<input type="checkbox" id="render-white-black-checkbox" />');
                    gsChck.on('change', function() {
                        window.renderConfig['white_on_black'] = $(this).prop('checked');
                    });
                    chckbxCont.append(gsChck);
                    chckbxCont.append($('<label for="render-white-black-checkbox">flip colors (white on black)</label>'));
                    markup.append(chckbxCont);
                }

                // append to UI if any controls allowed
                if(markup.children().length) {

                    if(showApplyButtons) {
                        let buttonContainer = $('<div style="float:right"></div>');

                        // button to apply
                        let applyBtn = $('<button class="btn btn-sm btn-primary" style="margin-right:5px">Apply</button>');
                        applyBtn.on('click', function() {
                            rerenderAll();
                        });
                        buttonContainer.append(applyBtn);

                        // button to reset to defaults (window.renderConfig_default)
                        let resetDefaultsBtn = $('<button class="btn btn-sm btn-warning">Reset</button>');
                        resetDefaultsBtn.on('click', function() {
                            JSON.parse(JSON.stringify(window.renderConfig_default));
                            // update GUI controls too
                            //TODO: implement case where not all adjustments are available
                            ['red', 'green', 'blue'].map((band) => {
                                $('#band-select-'+band).val(window.renderConfig['bands']['indices'][band].toString());
                            });
                            ['min', 'max'].map((valID) => {
                                let val = get_render_config_val(window.renderConfig, ['contrast', 'percentile', valID], 0.0);
                                $('#contrast-stretch-perc-'+valID).val(val);
                            });
                            let val = 128 + get_render_config_val(window.renderConfig, 'brightness', 0);
                            $('#render-brightness-slider').val(val)
                            $('#render-brightness-slider').trigger('input');    // to set percentage field

                            $('#render-grayscale-checkbox').prop('checked', get_render_config_val(window.renderConfig, 'grayscale', false));
                            $('#render-white-black-checkbox').prop('checked', get_render_config_val(window.renderConfig, 'white_on_black', false));

                            rerenderAll();
                        });
                        buttonContainer.append(resetDefaultsBtn);
                        markup.append(buttonContainer);
                    }

                    $(parentDiv).append(markup);
                } else {
                    $('#toolbox-tab-image').hide();  //TODO
                }
            } catch(error) {
                console.error(error);
            }
        }
    });
}