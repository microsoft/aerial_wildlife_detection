/*
    Functionality for monitoring and interacting connected AIWorkers
    from the client side.

    2019 Benjamin Kellenberger
*/


class AIWorkerJob {
    /*
        Wrapper for individual jobs submitted to the worker(s).
    */
    constructor(jobNumber, jobID, state) {
        this.jobNumber = jobNumber;
        this.jobID = jobID;
        this.jobType = jobNumber + ' &mdash; ' + state['type'];


        //TODO: we don't parse the submitted date to avoid time zone shifts
        this.timeCreated = new Date();

        // milliseconds for a single time step
        this.averageTimeRequired = 1000;
        this.numPolls = 0;

        this._setup_markup();
        this._parse_state(state);
    }

    _parse_state(state) {
        //TODO: error handling?
        this.type = state['type'];
        // this.timeCreated = Date.parse(state['submitted']);
        this.status = state['status'];
        
        // status message
        var message = 'running';
        if(status.hasOwnProperty('meta')  && status['meta'].hasOwnProperty('message')) {
            message = status['meta']['message'];
        }

        // progress
        if(this.status === 'PROGRESS') {
            this.total = parseInt(state['meta']['total']);
            this.current = Math.min(this.total, parseInt(state['meta']['done']));

            // update progress bar
            var newWidthPerc = 100*(this.current / this.total);
            this.progressBar.animate({
                'width': newWidthPerc + '%'
            }, 1000);

            // update message
            message += ' (' + this.current + '/' + this.total + ')';

        } else if(this.status === 'SUCCESS' || this.status === 'FAILURE') {
            // completed; hide progress bar
            this.progressBar.hide();

            // stop timers
            clearInterval(this.timerHandle);

        } else {
            // no progress status; set indeterminate progress bar
            this.progressBar.css('width', '100%');
        }

        // status message
        this.statusDiv.html(message);


        // update time counter (markup will automatically update)
        this.numPolls += 1;
        this.timeElapsed = new Date() - this.timeCreated;
        this.averageTimeRequired += this.timeElapsed / Math.max(1, this.current);
        this.timeRemaining = (this.total - this.current) * (this.averageTimeRequired / Math.max(1,this.numPolls));
    }

    _setup_markup() {
        this.markup = $('<div class="ai-worker-job expanded"></div>');

        // job type
        this.markup.append($('<h3 style="margin-left:12px;cursor:pointer">' + this.jobType + '</h3>'));

        // details container
        var detailsC = $('<div class="job-details"></div>');

        // status message
        var statusInd = $('<div></div>');
        statusInd.append($('<span>status: </span>'));
        this.statusDiv = $('<span></span>');
        statusInd.append(this.statusDiv);
        detailsC.append(statusInd);

        // progress indicator
        var prInd = $('<div></div>');
        var pbarWrapper = $('<div class="progressbar"></div>');
        prInd.append(pbarWrapper);
        this.progressBar = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar);

        // elapsed and remaining time
        var timeInd = $('<div class="time-indicator"></div>');
        this.elapsedTime = $('<div>00:00</div>');
        timeInd.append(this.elapsedTime);
        this.remainingTime = $('<div style="float:right">-00:00</div>');
        timeInd.append(this.remainingTime);
        prInd.append(timeInd);
        var self = this;
        this.timerHandle = setInterval(function() {
            self.timeElapsed += 1000;
            self.timeRemaining = Math.max(0, self.timeRemaining - 1000);
            $(self.elapsedTime).html(window.msToTime(self.timeElapsed));
            $(self.remainingTime).html('-'+window.msToTime(self.timeRemaining));
        }, 1000);

        detailsC.append(prInd);
        this.markup.append(detailsC);

        // collapsible
        this.markup.click(function() {
            if($(this).hasClass('expanded')) {
                detailsC.slideUp();
            } else {
                detailsC.slideDown();
            }
            $(this).toggleClass('expanded');
        });
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
        // this._query_worker_status(); //TODO
        this._query_task_status();
    }

    __toggle_panel() {
        if($('#ai-worker-panel').is(':visible')) {
            $('#ai-worker-panel').slideUp();
        } else {
            $('#ai-worker-panel').slideDown();
        }
    }

    _setup_markup() {

        // this.workersContainer = $('#ai-worker-entries');     //TODO
        this.taskContainer = $('#ai-tasks-entries');

        // mini-panel global progress indicator
        this.prInd = $('<div class="minipanel-progress" style="display:none"></div>');
        this.miniStatus = $('<span style="margin-right:10px"></span>');
        this.prInd.append(this.miniStatus);
        var pbarWrapper = $('<div class="progressbar" id="minipanel-progressbar"></div>');
        this.progressBar = $('<div class="progressbar-filler progressbar-active"></div>');
        pbarWrapper.append(this.progressBar);
        this.prInd.append(pbarWrapper);
        $('#ai-minipanel-status').append(this.prInd);

        // make panel collapsible
        var self = this;
        $('#ai-worker-minipanel').click(function() {
            self.__toggle_panel();
        });
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

                setTimeout(function() { self._query_worker_status(); }, 10000);   //TODO: make parameter
            },
            error: function(a,b,c) {
                //TODO
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
                // global task status
                var numTotal = 0;
                var numDone = 0;

                // parse tasks
                var tasks = data['status']['tasks'];
                
                for(var key in tasks) {
                    if(!self.tasks.hasOwnProperty(key)) {
                        var task = new AIWorkerJob(Object.keys(self.tasks).length+1, key, tasks[key]);
                        self.tasks[key] = task;
                        self.taskContainer.append(task.markup);

                    } else {
                        // update task
                        self.tasks[key].update_state(tasks[key]);
                    }

                    // parse task progress
                    if(tasks[key].hasOwnProperty('meta')) {
                        if(tasks[key]['meta'].hasOwnProperty('total')) {
                            numTotal += tasks[key]['meta']['total'];
                        }
                        if(tasks[key]['meta'].hasOwnProperty('done')) {
                            numDone += tasks[key]['meta']['done'];
                        }
                    }
                }

                // check for completed tasks (TODO: what to do with them? Completed history?)
                for(var key in self.tasks) {
                    if(!tasks.hasOwnProperty(key)) {
                        self.tasks[key].markup.remove();
                        delete self.tasks[key];
                    }
                }

                // update global progress bar and status message
                if(Object.keys(self.tasks).length > 0) {
                    $(self.prInd).show();
                    $(self.progressBar).show();
                    $(self.miniStatus).show();
                    var msg = Object.keys(self.tasks).length + ' tasks';
                    if(numTotal > 0) {
                        var newWidthPerc = 100*(numDone / numTotal);
                        self.progressBar.animate({
                            'width': newWidthPerc + '%'
                        }, 1000);
                        msg += ' (' + Math.round(newWidthPerc) + '%)';
                    } else {
                        // tasks going on, but progress unknown, set indeterminate
                        self.progressBar.animate({
                            'width': '100%'
                        }, 1000);
                    }
                    $(self.miniStatus).html(msg);
                } else {
                    // no tasks; hide progress bar and status message
                    $(self.prInd).hide();
                    $(self.progressBar).hide();
                    $(self.miniStatus).hide();
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