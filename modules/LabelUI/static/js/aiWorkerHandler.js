/*
    Functionality for monitoring and interacting connected AIWorkers
    from the client side.

    2019 Benjamin Kellenberger
*/


class AIWorkerJob {
    /*
        Wrapper for individual jobs submitted to the worker(s).
    */
    constructor(jobID, state) {
        this.jobID = jobID;

        this._setup_markup();
        this._parse_state(state);
    }

    _parse_state(state) {
        //TODO: error handling?
        this.type = state['type'];
        this.submitted = Date.parse(state['submitted']);
        this.status = state['status'];
        this.statusDiv.html(this.status);        //TODO: parse properly
        
        // progress
        if(this.status === 'PROGRESS') {
            this.current = parseInt(state['meta']['done']);
            this.total = parseInt(state['meta']['total']);

            // update progress bar
            var newWidthPerc = 100*(this.current / this.total);
            this.progressBar.animate({
                'width': newWidthPerc + '%'
            }, 1000);

        } else if(this.status === '') {
            // completed; hide progress bar
            this.progressBar.hide();

        } else {
            // no progress status; set indeterminate progress bar
            this.progressBar.css('width', '100%');
        }


        // time counter (TODO: set interval for counters)
        var timeCreated = Date.parse(state['submitted']);
        var timeElapsed = new Date() - timeCreated;
        var timeRemaining = (timeElapsed / this.current) * this.total - timeElapsed;

        this.elapsedTime.html(window.msToTime(timeElapsed));
        this.remainingTime.html(window.msToTime(timeRemaining));
    }

    _setup_markup() {
        this.markup = $('<div class="ai-worker-job"></div>');

        // status message
        var statusInd = $('<div></div>');
        statusInd.append($('<span>status:</span>'));
        this.statusDiv = $('<span></span>');
        statusInd.append(this.statusDiv);
        this.markup.append(statusInd);

        // progress indicator
        var prInd = $('<div></div>');
        var pbarWrapper = $('<div class="progressbar"></div>');
        prInd.append(pbarWrapper);
        this.progressBar = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar);

        // elapsed and remaining time
        var timeInd = $('<div></div>');
        this.elapsedTime = $('<div></div>');
        timeInd.append(this.elapsedTime);
        this.remainingTime = $('<div></div>');
        timeInd.append(this.remainingTime);
        prInd.append(this.remainingTime);
        this.markup.append(prInd);
    }

    update_state(state) {
        this._parse_state(state);
    }
}




class AIWorkerHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.tasks = {};
        this._setup_markup();
        this._query_worker_status();
        this._query_task_status();
    }

    _setup_markup() {
        this.parentDiv.append('<div style="font-size:14px">Worker status</div>');
        this.workersContainer = $('<div id="ai-worker-workers"><span id="ai-worker-workers-count"></span></div>');
        this.parentDiv.append(this.workersContainer);

        this.parentDiv.append('<div style="font-size:14px">Job status</div>');
        this.taskContainer = $('<div id="ai-worker-jobs"></div>');
        this.parentDiv.append(this.taskContainer);
    }


    _query_worker_status() {
        var self = this;
        $.ajax({
            url: window.aiControllerURI + 'status?workers=true',
            type: 'GET',
            success: function(data) {
                // parse workers
                var workers = data['status']['workers'];
                
                //TODO: simple count for now
                var count = Object.keys(workers).length;
                var suffix = (count == 1 ? '' : 's');
                $('#ai-worker-workers-count').html('connected to ' + count + ' worker' + suffix);

                // for(var key in workers) {
                //     console.log(key)
                // }

                setTimeout(function() { self._query_worker_status(); }, 10000);   //TODO
            },
            error: function(a,b,c) {
                console.log(a);
                console.log(b);
                console.log(c);
            }
        })
    }


    _query_task_status() {
        var self = this;
        $.ajax({
            url: window.aiControllerURI + 'status?tasks=true',
            type: 'GET',
            success: function(data) {
                // parse tasks
                var tasks = data['status']['tasks'];
                for(var key in tasks) {
                    if(!self.tasks.hasOwnProperty(key)) {
                        var task = new AIWorkerJob(key, tasks[key]);
                        self.tasks[key] = task;
                        self.taskContainer.append(task.markup);

                    } else {
                        // update task
                        self.tasks[key].update_state(tasks[key]);
                    }
                }

                // check for completed tasks (TODO: what to do with them? Completed history?)
                for(var key in self.tasks) {
                    if(!tasks.hasOwnProperty(key)) {
                        self.tasks[key].markup.remove();
                        delete self.tasks[key];
                    }
                }

                setTimeout(function() { self._query_task_status(); }, 5000);   //TODO
            },
            error: function(a,b,c) {
                console.log(a);
                console.log(b);
                console.log(c);
            }
        });
    }
}