/*
    Helper functions to poll for tasks dispatched
    through Celery.

    2020-22 Benjamin Kellenberger
*/

if(window.baseURL === undefined) window.baseURL = '';

function poll_status(taskID, successHandle, errorHandle, progressHandle, timeout) {
    /**
     * Polls the main server for tasks dispatched through Celery by the "DataAdministration" module.
     * Executes either the function specified under "successHandle" or "errorHandle" (if provided).
     * Repeats polling if the result is not (yet) ready, and if the task has not failed. If
     * "progressHandle" is a function, it will be called upon any status polling result that has
     * neither finished nor failed.
     *
     * Note that this function only polls the status for a single task with given ID.
     */
    var tHandle = undefined;
    function __do_poll() {
        return $.ajax({
            url: window.baseURL + 'pollStatus',
            method: 'POST',
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            data: JSON.stringify({'taskID': taskID}),
            success: function(data) {
                // check if result is ready
                data = data['response'][taskID];
                if(data.hasOwnProperty('result')) {
                    clearInterval(tHandle);
                    if(successHandle !== undefined && successHandle !== null) {
                        try {
                            return successHandle(data);
                        } catch {}
                    }
                } else if(data.hasOwnProperty('status') &&
                        data['status'].toLowerCase() === 'failure') {
                    clearInterval(tHandle);
                    if(errorHandle !== undefined && errorHandle !== null) {
                        try {
                            return errorHandle(data);
                        } catch {}
                    }
                } else if(typeof(progressHandle) === 'function') {
                    try {
                        return progressHandle(data);
                    } catch {}
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
                        return __do_poll(); //TODO: verify whether this causes infinite loops
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
