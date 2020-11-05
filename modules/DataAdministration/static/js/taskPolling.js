/*
    Helper functions to poll for tasks dispatched
    through Celery.

    2020 Benjamin Kellenberger
*/


function poll_status(taskID, successHandle, errorHandle, timeout) {
    /**
     * Polls the main server for tasks dispatched through Celery
     * by the "DataAdministration" module. Executes either the
     * function specified under "successHandle" or "errorHandle"
     * (if provided). Repeats polling if the result is not (yet)
     * ready, and if the task has not failed.
     */
    var tHandle = undefined;
    function __do_poll() {
        return $.ajax({
            url: 'pollStatus',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({'taskID': taskID}),
            success: function(data) {
                // check if result is ready
                if(data.hasOwnProperty('response') && data['response'].hasOwnProperty('result')) {
                    clearInterval(tHandle);
                    if(successHandle !== undefined && successHandle !== null) {
                        try {
                            return successHandle(data['response']['result']);
                        } catch {}
                    }
                } else if(data.hasOwnProperty('status') && data['status'].toLowerCase() === 'failure') {
                    clearInterval(tHandle);
                    if(errorHandle !== undefined && errorHandle !== null) {
                        try {
                            return errorHandle(data);
                        } catch {}
                    }
                }
            },
            error: function(data) {
                clearInterval(tHandle);
                if(errorHandle !== undefined && errorHandle !== null) {
                    try {
                        return errorHandle(data);
                    } catch {}
                }
            },
            statusCode: {
                401: function(xhr) {
                    return window.renewSessionRequest(xhr, function() {
                        return __do_poll();                 //TODO: verify whether this causes infinite loops
                    });
                }
            }
        });
    }

    if(timeout === undefined || timeout === null) {
        __do_poll();
    } else {
        tHandle = setInterval(__do_poll, 1000);
    }
}