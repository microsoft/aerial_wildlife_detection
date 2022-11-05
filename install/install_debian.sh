#!/bin/bash

# Installation routine for AIDE on Debian/Ubuntu systems.
#
# 2021-22 Benjamin Kellenberger


# -----------------------------------------------------------------------------
# FLAGS
# -----------------------------------------------------------------------------
config_file=                        # custom settings.ini file (will use config/settings.ini if empty)
python_exec=                        # alternative Python executable (will use default if empty)
aide_root=                          # directory of AIDE source code
append_environment_args=FALSE       # append the "AIDE_CONFIG_PATH" environment variable too user profile
with_gpu=TRUE                       # enable or disable installation of GPU libraries
pg_version=12                       # PostgreSQL version
tcp_keepalive=TRUE                  # set TCP/IP keepalive timer
install_daemon=                     # install systemd daemon
yes=FALSE                           # skip all confirmations
aide_group=aide                     # default group name for AIDE services
aide_daemon_user=aide_celery        # default user account for AIWorker
advanced_mode=FALSE                 # advanced mode: ask for credentials for every service if missing
test_only=FALSE                     # skip installation and only do checks and tests if true


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

# constants
INSTALLER_VERSION=2.1.210804
MIN_PG_VERSION=9.5
PG_KEY=ACCC4CF8.asc
DEFAULT_PORT_RABBITMQ=5672
DEFAULT_PORT_REDIS=6379
SYSTEMD_TARGET_SERVER=aide-server                   # for AIDE Web server: /etc/systemd/system/$(SYSTEMD_TARGET_SERVER).service
SYSTEMD_TARGET_WORKER=aide-worker                   # for AIWorker:        /etc/systemd/system/$(SYSTEMD_TARGET_WORKER).service
SYSTEMD_TARGET_WORKER_BEAT=aide-worker-beat         # for AIWorker (Celerybeat): /etc/systemd/system/$(SYSTEMD_TARGET_WORKER_BEAT).service
SYSTEMD_CONFIG_WORKER=/etc/default/celeryd_aide     # for AIWorker: config script for Celery daemon
TMPFILES_VOLATILE_DIR=/etc/tmpfiles.d/              # for setting permissions on volatile files required by Celery daemon

# modules that can be manually disabled (rest is determined by credentials)
install_labelUI=true
install_aiworker=true
if [ ${#AIDE_MODULES} -gt 0 ]; then
    # environment variable "AIDE_MODULES" found; extract from there
    if [[ $AIDE_MODULES != *"LabelUI"* ]]; then
        install_labelUI=false
    fi
    if [[ $AIDE_MODULES != *"AIWorker"* ]]; then
        install_aiworker=false
    fi
fi

if [ -f $AIDE_CONFIG_PATH ]; then
    # environment variable "AIDE_CONFIG_PATH" provided and file exists; extract from there
    config_file=$AIDE_CONFIG_PATH
fi

# argparse
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -c|--config_file|--config-file)
      config_file="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--python_exec|--python-exec)
      python_exec="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--celery_exec|--celery-exec)
      celery_exec="$2"
      shift # past argument
      shift # past value
      ;;
    -a|--advanced_mode|--advanced-mode)
      advanced_mode=true
      shift # past argument
      ;;
    -t|--test_only|--test-only)
      test_only=true
      shift # past argument
      ;;
    -y|--yes)
      yes=true
      shift # past argument
      ;;
    --config_file_out|--config-file-out)
      config_file_out="$2"
      shift # past argument
      shift # past value
      ;;
    --no_labelui|--no-labelui)
      install_labelUI=false
      shift # past argument
      ;;
     --no_aiworker|--no-aiworker)
      install_aiworker=false
      shift # past argument
      ;;
    --no_gpu|--no-gpu)
      with_gpu=false
      shift # past argument
      ;;
    --no_tcp_keepalive|--no-tcp-keepalive)
      tcp_keepalive=false
      shift # past argument
      ;;
    --append_environment_args|--append-environment-args)
      append_environment_args=true
      shift # past argument
      ;;
    --pg_version|--pg-version)
      pg_version="$2"
      shift # past argument
      shift # past value
      ;;
    --install_daemon|--install-daemon)
      install_daemon=true
      shift # past argument
      ;;
    --daemon_user|--daemon-user)
      aide_daemon_user="$2"
      shift # past argument
      shift # past value
      ;;
    -g|--group)
      aide_group="$2"
      shift # past argument
      shift # past value
      ;;
    -v|-V|--version)
      echo "install_debian $INSTALLER_VERSION"
      exit 0
      ;;
    -h|--help|?|help|HELP|*)    # unknown option; show help
      help_text=$(cat << EOF
\e[1m$BASH_SOURCE

\e[1mSYNOPSIS
    install_debian.sh\e[0m {-c | --config-file config_file} [--config-file-out config_file_out]
        {-a | --advanced-mode} {-y | --yes}
        {-p | --python-exec python_exec} {-c | --celery-exec celery_exec}
        [--install-daemon] [--daemon-user daemon_user] {-g | --group group}
        [--no-labelui] [--no-aiworker]
        [--pg-version pg_version]
        [--no-tcp-keepalive] [--append-environment-args]
        {-t | --test-only}
        {-h | --help}

\e[1mDESCRIPTION\e[0m
    Installer for AIDE, the Annotation Interface for Data-driven Ecology, on Debian systems (Debian, Ubuntu, etc.). Tested on Ubuntu 20.04 LTS.
    Performs the following operations (optional steps in square brackets):
    - installation of system-wide libraries and dependencies
    - installation of Python libraries and dependencies
    - installation and configuration of services: PostgreSQL server, RabbitMQ server, Redis server
    - [setup of systemd service for AIDE Web server and AIWorker]
    - tests of individual modules and service accesses

    To do so, the installer retrieves settings from a configuration file, either specified via argument "--config-file" or taken from the default
    (config/settings.ini) if empty. Required but missing parameters (hostnames, ports, passwords, etc.) will be queried to the user and written to the speci-
    fied config file, or else saved under a new name if desired (see argument "--config-file-out").

    This installer can run in two modes:
    - interactive mode (default), accepting user inputs and asking for confirmation
    - automated mode, which skips confirmation inputs (flag "-y"). Note that this mode is only fully automatic if every required parameter is present in the
      specified configuration file ("--config-file"). Otherwise the installer will ask for user input.

\e[1mOPTIONS\e[0m
    \e[1m-c\e[0m, \e[1m--config-file\e[0m \e[4mfile\e[0m
            Path (relative or absolute) to the configuration (*.ini) file to use as a basis. If not specified the installer will first check for environment
            variable "AIDE_CONFIG_PATH" and will use this if specified. If environment variable is unset, it will use the default "config/settings.ini" file.
    \e[1m--config-file-out\e[0m \e[4mfile\e[0m
            Path (relative or absolute) to save the configuration (*.ini) file to in case it gets modified during the installation process. If not specified
            the installer will ask for a location during the process unless flag "-y" is set, in which case the original file gets overwritten. Note that
            configuration (*.ini) file modifications result in any comments being discarded from the file.
    \e[1m-a\e[0m, \e[1m--advanced-mode\e[0m
            Turn on advanced mode. In this mode any missing credentials or server locations (host, port, etc.) will be prompted to the user during installation.
            If disabled (default), missing credentials will be substituted with default values (e.g., "localhost" for hostnames) or auto-generated (for service
            credentials). Note that the AIDE admin user account name, E-mail address and password will never be auto-generated and will be prompted unless al-
            ready specified in the configuration (*.ini) file.
    \e[1m-y\e[0m, \e[1m--yes\e[0m
            Skip asking for confirmation. Note that if the configuration (*.ini) file is incomplete w.r.t. credentials and server location values the installer
            may still ask for inputs.
    \e[1m-p\e[0m, \e[1m--python-exec\e[0m \e[4mfile\e[0m
            Custom location of the Python executable to be used. Note that this determines the Python environment the required packages are installed to.
            Uses the current Python executable ("which python") by default. Specify together with "--celery-exec" if needed to install AIDE into a different
            environment. Usually, a better way to install AIDE is to activate the custom Python environment prior to launching the installation script.
    \e[1m-p\e[0m, \e[1m--celery-exec\e[0m \e[4mfile\e[0m
            Custom location of the Celery executable to be used. Uses the current Celery executable ("which celery") by default. Specify together with
            "--python-exec" if needed to install AIDE into a different environment.
    \e[1m--install-daemon\e[0m
            Set flag to set up systemd daemon processes for AIDE to launch it e.g. at system start. Depending on the configuration, this creates up to three
            daemon processes:
                aide-server.service:        for the Web frontend (if enabled)
                aide-worker.service:        for the AIWorker (if enabled)
                aide-worker-beat.service:   periodic worker for the FileServer to check for modified images (if enabled)
            If unset, the installer will ask whether processes should be installed. If flag "-y" specified, installer will automatically set up the daemon
            processes.
    \e[1m--daemon-user\e[0m \e[4muser\e[0m
            User account name to employ to run systemd daemon processes. User account will be created with given name if necessary. Defaults to "aide_celery".
    \e[1m-g\e[0m, \e[1m--group\e[0m \e[4mgroup\e[0m
            Group for systemd daemon processes. Note that both the daemon user and current account running this installer will be assigned to this group
            and that permissions and ownership for important files (configuration *.ini file, other configuration files) will be set accordingly.
            Defaults to "aide".
    \e[1m--no-labelui\e[0m
            Disable installation of LabelUI Web server. This affects the systemd daemon process and "AIDE_MODULES" environment variable. Use if you wish to
            branch out the installation of AIDE to multiple machines.
            If not explicitly specified, the installer will determine whether to install the LabelUI component based on the presence of the string "LabelUI"
            in the environment variable "AIDE_MODULES". Defaults to true if environment variable is unset.
    \e[1m--no-aiworker\e[0m
            Disable installation of AIWorker. This affects the systemd daemon process and "AIDE_MODULES" environment variable. Use if you wish to
            branch out the installation of AIDE to multiple machines.
            If not explicitly specified, the installer will determine whether to install the AIWorker component based on the presence of the string "AIWorker"
            in the environment variable "AIDE_MODULES". Defaults to true if environment variable is unset.
    \e[1m--pg-version\e[0m \e[4mversion\e[0m
            PostgreSQL server version to install. Must be >= 9.5; defaults to 12.
    \e[1m--no-tcp-keepalive\e[0m
            Disable TCP keepalive timer settings for system. Keepalive timers are set by default in file "/etc/sysctl.conf" to prevent connection drops to
            PostgreSQL server.
    \e[1m--append-environment-args\e[0m
            Set environment variables "AIDE_CONFIG_PATH" and "AIDE_MODULES" in "${HOME}/.profile" for easier manual launching of AIDE. Disabled by default.
    \e[1m-t\e[0m, \e[1m--test-only\e[0m
            Skip installation and just perform tests. Note that this does not disable prompting for missing variables and modifying the configuration (*.ini)
            file.

  \e[1mGetting help\e[0m
    \e[1m-h\e[0m, \e[1m--help\e[0m
            Display this help information.
    \e[1m-V\e[0m, \e[1m--version\e[0m
            Display installer version.

\e[1mEXIT STATUS\e[0m
    \e[1m0\e[0m Successful program execution.

    \e[1m1\e[0m Attempt to run installer from root (installer should be run from regular user account with administrative privileges).

    \e[1m2\e[0m Attempt to run installer from account without administrative privileges.

    \e[1m3\e[0m Attempt to run installer from non-Linux operating system.

    \e[1m4\e[0m Missing package manager "aptitude" ("apt", "apt-get").

    \e[1m5\e[0m Outdated Python executable specified. Make sure to run installer in Python environment with Python version 3.5 or greater.

    \e[1m6\e[0m Cannot find AIDE installation directory. Make sure to leave this installer in the original place in the AIDE code base.

    \e[1m7\e[0m Outdated PostgreSQL server found (version < 9.5).

    \e[1m8\e[0m Remote PostgreSQL server cannot be contacted. Make sure current machine and account have access permissions to database and server.

\e[1mHISTORY\e[0m
    Apr 12, 2022: Various bug fixes on PostgreSQL installation and daemonization routines; made installer compatible with exotic account and machine names.
    Aug 4, 2021: Initial installer release by Benjamin Kellenberger (benjamin.kellenberger@epfl.ch)

$INSTALLER_VERSION                  https://github.com/microsoft/aerial_wildlife_detection              
EOF
)
      echo -e "$help_text"
      exit 0
      ;;
  esac
done

# utility functions
getBool() {
    case $1 in
        true) echo true ;;
        false) echo false ;;
        TRUE) echo true ;;
        FALSE) echo false ;;
        YES) echo true ;;
        NO) echo false ;;
        *) echo false ;;
    esac
}

log() {
    msg="$1"
    args=
    space=''
    space_log=''
    if [[ "$(getBool $3)" == true ]]; then
        # no newline, but add trailing white spaces to text for "[ OK ]" flag at the end
        width=$(expr $(tput cols) - 6 - ${#msg})
        space="$(printf '%*s' $width)"
        width_log=$(expr 74 - ${#msg})
        space_log="$(printf '%*s' $width_log)"
        args="-n"
    fi
    quiet="$(getBool $2)"
    if [[ $quiet == false ]]; then
        echo $args -e "$msg$space"
    fi
    echo $args -e "$msg$space_log" | sed "s/\x1B\[\([0-9]\{1,2\}\(;[0-9]\{1,2\}\)\?\)\?[mGK]//g" >> $logFile
}

warn() {
    log "\e[1m[\e[33mWARNING]\e[0m $1"
    if [ $yes ]; then
        echo "TRUE"
        return
    elif [[ $(getBool $2) == true ]]; then
        log "\nContinue?"
        while true; do
            read -p "[yes/no]" yn
            case $yn in
                [Yy]* ) log "[yes/no] $yn" && echo "TRUE" && break;;
                [Nn]* ) log "[yes/no] $yn" && echo "FALSE" && break;;
            esac
        done
    fi
}

abort() {
    errorCode=$1
    errorMsg=$2
    if [ ${#errorMsg} -eq 0 ]; then
        errorMsg="An unknown error occurred"
    fi
    log "\e[1m\e[31m[ERROR $errorCode]\e[0m $errorMsg. Aborting..."
    log "Log written to file '$logFile'."
    exit $errorCode
}

# log file
logFile="install_debian_$(date +'%Y%m%d_%H_%M_%S').log"
# echo -e "AIDE installation log from $(date)\n" >> $logFile

# splash screen
python <<EOF
print(f'''\033[96m
#################################       
                                        installer version $INSTALLER_VERSION
   ###    #### ########  ########       
  ## ##    ##  ##     ## ##             
 ##   ##   ##  ##     ## ##             
##     ##  ##  ##     ## ######  
#########  ##  ##     ## ##      
##     ##  ##  ##     ## ##             
##     ## #### ########  ######## 

#################################\033[0m
''')
EOF

log "AIDE version $AIDE_VERSION; installer version $INSTALLER_VERSION." "TRUE"

# config file
config_file_modified=0

getConfigParam() {
    echo $($python_exec <<EOF
import configparser
c=configparser.ConfigParser()
try:
    c.read('$config_file')
    print(c['$1']['$2'])
except:
    print('')
EOF
)
}

saveConfigParam() {
    $python_exec <<EOF
import configparser
c=configparser.ConfigParser()
c.read('$config_file_out')
c['$1']['$2'] = '$3'
with open('$config_file_out', 'w') as cf:
    c.write(cf)
EOF
}

promptValue() {
    val=$3;
    if [[ ${#val} -gt 0 ]]; then
        # value provided; return directly
        echo $val
        return 0
    fi
    silent=$(getBool $2)
    defaultVal=$4
    defaultText=
    if [ ${#defaultVal} -gt 0 ]; then
        if [[ $advanced_mode == false ]]; then
            # advanced mode disabled; return default value directly
            echo $defaultVal
            return 1
        fi
        defaultText=" (leave empty for default: '$defaultVal')"
    fi
    ask=true
    while [[ $ask == true ]]; do
        if [[ $silent == true ]]; then
            read -s -p "Please provide a value for unset parameter '$1'$defaultText: " inp
        else
            read -p "Please provide a value for unset parameter '$1'$defaultText: " inp
        fi
        val=$inp
        if [[ ${#defaultVal} -gt 0 ||  ${#val} -gt 0 ]]; then
            if [[ ${#defaultVal} -gt 0 && ${#val} -eq 0 ]]; then
                val=$defaultVal
            fi
            ask=false
        fi
    done
    if [[ $silent == true ]]; then
        log "Please provide a value for unset parameter '$1'$defaultText: ****" true
    else
        log "Please provide a value for unset parameter '$1'$defaultText: $val" true
    fi
    echo $val
    return 1
}

getOrPromptValue() {
    vname=$1
    val=${!vname}
    modified=0
    if [ ${#val} == 0 ]; then
        val=$(getConfigParam "$2" "$3");
        while [ ${#val} == 0 ]; do
            val=$(promptValue "$vname" "$4" "$5" "$6")
            modified=1
        done
    fi
    echo $val
    return $modified
}

parseURL() {
    url=$($python_exec <<EOF
from urllib.parse import urlparse
o = urlparse('$1')
netloc = o.netloc.split('@')
if len(netloc) > 1:
  cred = netloc[0].split(':')
  user = cred[0]
  passwd = (cred[1] if len(cred)>1 else '')
  netloc = netloc[1]
else:
  user, passwd = '', ''
  netloc = netloc[0]
netloc = netloc.split(':')
port = (netloc[1] if len(netloc)>1 else '')
netloc = netloc[0]
path = o.path
if path.startswith('/'):
  path = path[1:]
print(f'{o.scheme}?{user}?{passwd}?{netloc}?{port}?{path}')
EOF
)
    echo $url
}

assembleURL() {
    scheme=$2

    if [ $# -eq 7 ]; then
        # all args supplied
        user=$3
        pass=$4
        netloc=$5
        port=$6
        path=$7
    elif [ $# -eq 6 ]; then
        # no user password
        user=$3
        pass=
        netloc=$4
        port=$5
        path=$6
    else
        # no authentication
        user=
        pass=
        netloc=$3
        port=$4
        path=$5
    fi
    authStr=
    if [ ${#user} -gt 0 ]; then
        authStr=$user
        if [ ${#pass} -gt 0 ]; then
            hidePass=$(getBool $1)
            if [[ $hidePass == true ]]; then
                pass='****'
            fi
            authStr="$authStr:$pass"
        fi
        authStr="$authStr@"
    fi
    if [ ${#netloc} -eq 0 ]; then
        # no netloc; assume localhost
        netloc='localhost'
    fi
    if [ ${#port} -gt 0 ]; then
        netloc="$netloc:$port"
    fi
    if [ ${#path} -gt 0 ]; then
        if [[ $path != /* ]]; then
            path="/$path"
        fi

        netloc="$netloc$path"
    fi
    echo "$scheme://$authStr$netloc"
}

isLocalhost() {
    # machine's IP addresses
    ips=($(curl -s -4 ifconfig.co))
    for i in "${ips[@]}"
    do
        if [[ "$i" == "$1" ]]; then
            echo true;
            return;
        fi
    done

    case $1 in
        $ips) echo true ;;
        localhost) echo true ;;
        127.0.0.1) echo true ;;
        $HOSTNAME) echo true ;;
        '/') echo true ;;
        '') echo true ;;
        *) echo false ;;
    esac
}

genVal() {
    if [[ "$1" == "password" ]]; then
        # password
        echo "$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c16)"
    else
        # user name, with a bit of random characters for minor extra protection
        randStr="$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c4)"
        echo "aide_${HOSTNAME}_${randStr}"
    fi
}

testPythonGPUaccess() {
    echo $($python_exec <<EOF
import sys
import torch
if not torch.cuda.is_available():
    print('CUDA not available from within Torch')
    sys.exit(0)
try:
    out=torch.rand(42).cuda()
except Exception as e:
    print(f'Error copying torch.Tensor to GPU (message: "{str(e)}")')
    sys.exit(0)
EOF
)
}

testAIWorkerAccess() {
    echo $($python_exec <<EOF
import sys
import modules
from util import celeryWorkerCommons
try:
    details = celeryWorkerCommons.getCeleryWorkerDetails()
    details = '\n'.join([f'name: {k}, info: {details[k]}' for k in details])
    print(f'{str(details)}')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {str(e)}')
    sys.exit(0)
EOF
)
}

# installation parameters
with_gpu=$(getBool $with_gpu)
tcp_keepalive=$(getBool $tcp_keepalive)
append_environment_args=$(getBool $append_environment_args)
yes=$(getBool $yes)
test_only=$(getBool $test_only)
advanced_mode=$(getBool $advanced_mode)



# -----------------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# -----------------------------------------------------------------------------

log "\e[1m[01/11] \e[36mChecking system...\e[0m"

# check if user is non-root and has admin privileges
if [ "$EUID" -eq 0 ]; then
    #Allow root install under New user
    read  -p  "Please provide a value for new user name: " inp1    
    adduser $inp1
    usermod -aG sudo $inp1
    su $inp1
fi
hasSudo=$(sudo -v)
if [ ${#hasSudo} -ne 0 ]; then
    abort 2 "Please run installer from a user with administrative privileges"
fi

# check system type
sysType="$(uname -s)";
if [ $sysType != "Linux" ]; then
    abort 3 "Host operating system is not a Linux distribution ('$sysType')"
fi

# check Linux distribution
distr="$(cat /etc/os-release)"      #TODO: parse output (NAME, VERSION_ID)

# check if aptitude is installed
apt_exec=$(command -v apt);
if [ "${#apt_exec}" -eq 0 ]; then
    abort 4 "Package manager ('apt', 'aptitude') not found"
fi

# Python version
if [ ${#python_exec} -eq 0 ]; then
    python_exec=$(command -v python);
fi
pyVer="$($python_exec --version 2>&1)"
if [[ $pyVer != Python\ 3* ]]; then
    abort 5 "Python executable '$python_exec' is incompatible ('$pyVer' != 'Python 3.*')"
fi
if [[ $python_exec == '/usr/bin/python' ]]; then
    proceed=$(warn "System Python executable specified ('$python_exec'); this is absolutely NOT recommended!" "TRUE")
    if [ !$(getBool proceed) ]; then
        log "Aborted."
        exit;
    fi
fi

# check if run from AIDE root
if [ ${#aide_root} -eq 0 ]; then
    aide_root="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )")"
fi
if [ ! -f $aide_root/constants/version.py ]; then
    abort 6 "Invalid directory specified for AIDE ('$aide_root') - did you move the installation script somewhere else?"
fi

# check for config file
if [[ ${#config_file} -gt 0 && -f $config_file ]]; then
    # config file specified; check for validity
    log "Existing config file ('$config_file') found."
else
    # no config file specified or found; use default
    config_file=$aide_root/config/settings.ini
    log "Using configuration file in '$config_file' as basis."
fi

# check for CUDA capable GPU
if [[ $with_gpu == true ]]; then
    if [ $(command -v 'nvidia-smi') ]; then
        gpuInfo="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
        echo "GPU(s) found:" | tee -a $logFile
        echo "$gpuInfo" | tee -a $logFile
    else
        gpuInfo=
        warn "WARNING: argument '--with-gpu' specified but no CUDA-capable GPU and/or driver found. Continuing with CPU-only libraries."
        with_gpu=false
    fi
else
    log "Installing CPU-only libraries."
fi



# -----------------------------------------------------------------------------
# PARSE PARAMETERS
# -----------------------------------------------------------------------------

log "\e[1m[02/11] \e[36mParsing parameters...\e[0m"

# extract parameters from config file, prompt for missing ones where necessary
adminName=$(getOrPromptValue "adminName" "Project" "adminName");
config_file_modified=$(($config_file_modified + $?))
adminEmail=$(getOrPromptValue "adminEmail" "Project" "adminEmail");
config_file_modified=$(($config_file_modified + $?))
adminPassword=$(getOrPromptValue "adminPassword" "Project" "adminPassword" "TRUE");
config_file_modified=$(($config_file_modified + $?))
echo;
if [[ $advanced_mode == false ]]; then
    log "Advanced mode disabled; any remaining parameters missing from configuration file will be substituted with defaults or auto-generated if possible."
fi
serverHost=$(getOrPromptValue "server_host" "Server" "host" "FALSE" "localhost");
config_file_modified=$(($config_file_modified + $?))
serverPort=$(getOrPromptValue "server_port" "Server" "port" "FALSE" "8080");
config_file_modified=$(($config_file_modified + $?))

# database
dbName=$(getOrPromptValue "database_name" "Database" "name" "FALSE" "aide");
config_file_modified=$(($config_file_modified + $?))
dbHost=$(getOrPromptValue "database_host" "Database" "host" "FALSE" "localhost");
config_file_modified=$(($config_file_modified + $?))
dbPort=$(getOrPromptValue "database_port" "Database" "port" "FALSE" "5432");
config_file_modified=$(($config_file_modified + $?))

dbUser=$(getConfigParam "Database" "user");
if [ ${#dbUser} -eq 0 ]; then
    if [[ $advanced_mode == true ]]; then
        dbUser=$(getOrPromptValue "database_user_name" "Database" "FALSE" "user");
    else
        dbUser=$(genVal "user")
    fi
    config_file_modified=1
fi
dbPassword=$(getConfigParam "Database" "password");
if [ ${#dbPassword} -eq 0 ]; then
    if [[ $advanced_mode == true ]]; then
        dbPassword=$(getOrPromptValue "database_user_password" "Database" "password" "TRUE");
        echo;
    else
        dbPassword=$(genVal "password")
    fi
    config_file_modified=1
fi

# AIController
aiController_uri=$(getOrPromptValue "AIController_uri" "Server" "aiController_uri" "FALSE" "localhost");

# FileServer
dataServer_uri=$(getOrPromptValue "FileServer_uri" "Server" "dataServer_uri" "FALSE" "/");

# message brokers
url="$(parseURL $(getConfigParam "AIController" "broker_URL"))";
IFS='?' read -r -a rabbitmqCred <<< $url
rabbitmqCred[0]="amqp"
if [ ${#rabbitmqCred[1]} -eq 0 ]; then
    if [[ $advanced_mode == true ]]; then
        rabbitmqCred[1]=$(promptValue "rabbitmq_user_name" false );
    else
        rabbitmqCred[1]=$(genVal "user")
    fi
    config_file_modified=1
fi
if [ ${#rabbitmqCred[2]} -eq 0 ]; then
    if [[ $advanced_mode == true ]]; then
        rabbitmqCred[2]=$(promptValue "rabbitmq_user_password" true );
        echo;
    else
        rabbitmqCred[2]=$(genVal "password")
    fi
    config_file_modified=1
fi
rabbitmqCred[3]=$(promptValue "rabbitmq_host" false ${rabbitmqCred[3]} "localhost");
config_file_modified=$(($config_file_modified + $?))
rabbitmqCred[4]=$(promptValue "rabbitmq_port" false ${rabbitmqCred[4]} "5672");
config_file_modified=$(($config_file_modified + $?))
rabbitmqCred[5]=$(promptValue "rabbitmq_path" false ${rabbitmqCred[5]} "aide_vhost");
config_file_modified=$(($config_file_modified + $?))

url="$(parseURL $(getConfigParam "AIController" "result_backend"))";
IFS='?' read -r -a redisCred <<< $url
redisCred[0]="redis"
redisCred[3]=$(promptValue "redis_host" false ${redisCred[3]} "localhost");
config_file_modified=$(($config_file_modified + $?))
redisCred[4]=$(promptValue "redis_port" false ${redisCred[4]} "6379");
config_file_modified=$(($config_file_modified + $?))
redisCred[5]=$(promptValue "redis_path" false ${redisCred[5]} "0");
config_file_modified=$(($config_file_modified + $?))

# check which modules are on this machine
is_database=$(isLocalhost $dbHost)
is_rabbitmq=$(isLocalhost ${rabbitmqCred[3]})
is_redis=$(isLocalhost ${redisCred[3]})
is_fileserver=$(isLocalhost $dataServer_uri)

# modules to install
install_aicontroller=$(isLocalhost $aiController_uri)
install_database=$(isLocalhost $dbHost)
install_rabbitmq=$(isLocalhost ${rabbitmqCred[3]})
install_redis=$(isLocalhost ${redisCred[3]})
install_fileserver=$(isLocalhost $dataServer_uri)

aide_modules=
if $install_labelUI ; then
    aide_modules="LabelUI"
fi
if $install_aicontroller ; then
    aide_modules="$aide_modules,AIController"
fi
if $install_aiworker ; then
    aide_modules="$aide_modules,AIWorker"
fi
if $install_fileserver ; then
    aide_modules="$aide_modules,FileServer"
fi
aide_modules="$(echo $aide_modules | sed 's/^,//g')"

# file server dir
if $install_fileserver ; then
    fsDir=$(getOrPromptValue "file_server_folder" "FileServer" "staticfiles_dir" "FALSE" "/srv/$aide_group/files")
    config_file_modified=$(($config_file_modified + $?))
fi



# -----------------------------------------------------------------------------
# LIST PARAMETERS
# -----------------------------------------------------------------------------

log "\e[1m[03/11] \e[36mParameters found as follows:\e[0m"

log "\e[1mInstallation arguments:\e[0m"
log "\tInput config file:                         $config_file"
log "\tPython path:                               $python_exec"
log "\tAIDE path:                                 $aide_root"
# log "\tInstall libraries with GPU support:        $with_gpu"          #TODO: currently doesn't change much
if [[ $install_database == true ]]; then
    log "\tPostgreSQL version:                        $pg_version"
fi
log "\tSet TCP keepalive timer for system:        $tcp_keepalive"
log "\tAppend environment args to user profile:   $append_environment_args"

log "\n\e[1mAIDE modules to be installed on this system:\e[0m"
log "\tLabelUI:       $install_labelUI"
log "\tDatabase:      $install_database"
log "\tRabbitMQ:      $install_rabbitmq"
log "\tRedis:         $install_redis"
log "\tFile server:   $install_fileserver"


log "\n\e[1mPaths and URLs:\e[0m"
log "\tDatabase:          $(assembleURL true 'postgresql' $dbUser $dbPassword $dbHost $dbPort $dbName )"
log "\tRabbitMQ:          $(assembleURL true ${rabbitmqCred[@]})"
log "\tRedis:             $(assembleURL true ${redisCred[@]})"
log "\tFile server dir:   $fsDir"

#TODO: also allow to make modifications (e.g. to modules to be installed)

# check if configuration modified
if [[ $config_file_modified -gt 0 ]]; then
    if [[ ${#config_file_out} -eq 0 ]]; then
        # no config_file_out specified
        if [[ $yes == true ]]; then
            # modify config_file in-place
            config_file_out=$config_file
        else
            # ask for confirmation
            log "\n\e[1m*** Configuration has been modified. ***\e[0m\nWould you like to save updates to original file ('$config_file')?"
            log "Please note that all comments in the file will be discarded in this process."
            log "Alternatively, you can specify a path to save a modified version of the configuration to:"
            query=true
            while [[ $query == true ]]; do
                read -p "[config file path (leave empty to overwrite existing)]" inp
                if [[ ${#inp} -eq 0 ]]; then
                    # modify config_file in-place
                    log "\tModifying config file '$config_file' in-place and discarding comments."
                    config_file_out=$config_file
                    query=false
                else
                    # check if new file exists
                    if [ -f "$inp" ]; then
                        log "File '$inp' already exists; please specify a new name for it."
                    else
                        # copy original file over
                        config_file_out=$inp
                        if [ $config_file != $config_file_out ]; then
                            cp $config_file $config_file_out
                        fi
                        log "\tSaving changes to file '$config_file_out' and using it for AIDE installation."
                        query=false
                    fi
                fi
            done
        fi
    fi
    log "Modifications to configuration by installer (including passwords) will be written to file '$config_file_out'."
else
    # no changes made; use original config file directly
    config_file_out=$config_file
    log "Using unmodified configuration file '$config_file'."
fi
# turn relative config file path into absolute
config_file_out="$(cd "$(dirname "$config_file_out")"; pwd -P)/$(basename "$config_file_out")"

# modify config file and prepend disclaimer
if [[ $config_file_modified -gt 0 ]]; then
    log "Updating configuration file..."

    saveConfigParam "Project" "adminName" $adminName
    saveConfigParam "Project" "adminEmail" $adminEmail
    saveConfigParam "Project" "adminPassword" $adminPassword
    saveConfigParam "Database" "name" $dbName
    saveConfigParam "Database" "host" $dbHost
    saveConfigParam "Database" "port" $dbPort
    saveConfigParam "Database" "user" $dbUser
    saveConfigParam "Database" "password" $dbPassword
    saveConfigParam "AIController" "broker_URL" $(assembleURL false ${rabbitmqCred[@]})
    saveConfigParam "AIController" "result_backend" $(assembleURL false ${redisCred[@]})
    saveConfigParam "FileServer" "staticfiles_dir" $fsDir

    echo "; Configuration file modified by script 'install_debian.sh' on $(date)" >> /tmp/install_debian_config_file
    echo "; from original file '$config_file'" >> /tmp/install_debian_config_file
    echo ";" >> /tmp/install_debian_config_file
    cat $config_file_out >> /tmp/install_debian_config_file
    mv /tmp/install_debian_config_file $config_file_out
fi

if ! $yes ; then
    task="installation"
    if $test_only ; then
        task="tests"
    fi
    log "\nProceed with $task?"
    while true; do
        read -p "[yes/no]" yn
        case $yn in
            [Yy]* ) log "$yn" && break;;
            [Nn]* ) log "Aborted." && exit;;
        esac
    done
fi


if ! $test_only ; then
    log "Starting installation..."
fi



# -----------------------------------------------------------------------------
# INSTALL COMMON DEPENDENCIES
# -----------------------------------------------------------------------------

log "\e[1m[04/11] \e[36mInstalling dependencies...\e[0m"
if $test_only ; then
    log "Skipping..."
else
    sudo apt-get update | tee -a $log;
    sudo apt-get install -y build-essential wget libpq-dev python-dev ffmpeg libsm6 libxext6 python3-opencv python3-pip | tee -a $log;
    pip install -U -r $aide_root/requirements.txt | tee -a $log;
fi

export AIDE_CONFIG_PATH=$config_file_out;
export AIDE_MODULES=$aide_modules;
export PYTHONPATH=$aide_root;



# -----------------------------------------------------------------------------
# INSTALL DATABASE
# -----------------------------------------------------------------------------

log "\e[1m[05/11] \e[36mDatabase...\e[0m"
if [[ $install_database == true ]]; then

    if ! $test_only ; then
        sudo apt-get install -y postgresql-client | tee -a $log;
    fi

    # check if psql and Postgres server already installed (and fetch version)
    psqlV=-1
    psqlInstalled=false
    psql_exec=$(command -v psql);
    postgresInstalled=false
    postgres_info="$(pgrep -u postgres -fa -- -D)"
    if [ "${#psql_exec}" -gt 0 ]; then
        psqlV=$(psql -V)
        psqlV=$(echo $psqlV | grep -o -E '[0-9]+\.[0-9]* ' | bc);
        log "Existing PostgreSQL client version $psqlV found."
        psqlInstalled=true
    fi
    if [ "${#postgres_info}" -gt 0 ]; then
        postgresInstalled=false
        log "Postgres account found."
    fi
    if [[ $psqlInstalled == true && $postgresInstalled == true ]]; then
        #TODO: try to connect
        install_database=false
    fi

    # postgresql.conf file
    pgConfFile="/etc/postgresql/$pg_version/main/postgresql.conf"

    if [[ $test_only == false && $install_database == true ]]; then
        log "Installing PostgreSQL server version $pg_version..."

        # install and configure PostgreSQL
        echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
        wget --quiet -O - https://www.postgresql.org/media/keys/$PG_KEY | sudo apt-key add -
        wget --quiet -O - https://www.postgresql.org/repos/apt/$PG_KEY | sudo apt-key add -
        sudo apt-get update && sudo apt-get install -y postgresql-$pg_version | tee -a $log;

        # modify authentication
        dbAuth=$dbUser
        if [[ ! $install_labelUI || ! $install_aicontroller || ! $install_fileserver || ! $install_aiworker ]]; then
            # at least one AIDE module is not on this host; enable access to others
            dbAuth="all"
            log "PostgreSQL authentication set to 'all'. If you know the host names of the machines you want to install other AIDE services on you may want to modify the authentication for extra security in file '/etc/postgresql/$pg_version/main/pg_hba.conf'."
        fi 

        sudo sed -i "s/\s*#\s*listen_addresses\s=\s'localhost'/listen_addresses = '\*'/g" $pgConfFile
        echo "host    $dbName             $dbAuth             0.0.0.0/0               md5" | sudo tee -a /etc/postgresql/$pg_version/main/pg_hba.conf > /dev/null

        # modify port
        sudo sed -i "s/\s*port\s*=\s*[0-9]*/port = $dbPort/g" $pgConfFile

        # restart PostgreSQL daemon
        sudo service postgresql restart
        sudo systemctl enable postgresql

        # start cluster
        sudo systemctl start postgresql@$pg_version-main            #TODO: needed?
        sudo pg_ctlcluster $pg_version main start
    else
        # check version
        if (( $(echo "$psqlV < $MIN_PG_VERSION" |bc -l) )); then
            abort 7 "Outdated PostgreSQL database found ($psqlV < 9.5). You will not be able to use this version with AIDE.\nPlease manually upgrade to a newer version (9.5 or greater)."
        else
            # check port
            pgConfFile="$(sudo -u postgres psql -c 'SHOW config_file' | grep -o -E '.*\.conf')";
            pgPort="$(cat $pgConfFile | grep -o -E '\s*port\s*=\s[0-9]*' | tr -dc '0-9')"
            if (( $dbPort != $pgPort )); then
                warn "Existing PostgreSQL database is configured to listen to different port than specified ($pgPort != $dbPort).\nInstallation will continue with currently set port $pgPort."                echo "WARNING: existing PostgreSQL database is configured to listen to different port than specified ($pgPort != $dbPort)." | tee -a $logFile
                dbPort=$pgPort
            fi
        fi
    fi

    # setup database
    if ! $test_only ; then
        log "Creating user..."
        sudo -u postgres psql -c "CREATE USER \"$dbUser\" WITH PASSWORD '$dbPassword';" | tee -a $logFile
        log "Creating database..."
        sudo -u postgres psql -c "CREATE DATABASE \"$dbName\" WITH OWNER \"$dbUser\" CONNECTION LIMIT -1;" | tee -a $logFile
        log "Granting connection to database..."
        sudo -u postgres psql -c "GRANT CREATE, CONNECT ON DATABASE \"$dbName\" TO \"$dbUser\";" | tee -a $logFile
        log "Creating UUID extension..."
        sudo -u postgres psql -d $dbName -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" | tee -a $logFile
        log "Granting table privileges..."
        sudo -u postgres psql -d $dbName -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO \"$dbUser\";" | tee -a $logFile

        log "Initializing AIDE database schema..."
        python $aide_root/setup/setupDB.py | tee -a $logFile
    fi

else
    # database is non-localhost; try to contact it
    log "Database is on remote server; attempting to contact..." "FALSE" "TRUE"
    pgStatus=$(pg_isready -U $dbUser -h $dbHost -p $dbPort -d $dbName)
    pgActive=$(pgStatus | grep "accepting connections")
    if [ ! ${#pgActive} ]; then
        log "\e[31m[FAIL]\e[0m"
        abort 8 "Failed to contact database server (message: '$pgStatus'). Please make sure remote server (postgres://$dbHost:$dbPort/$dbName) is reachable by $dbUser via md5 authentication."
    else
        log "\e[32m[ OK ]\e[0m"
    fi
fi



# -----------------------------------------------------------------------------
# INSTALL RABBITMQ
# -----------------------------------------------------------------------------

log "\e[1m[06/11] \e[36mRabbitMQ...\e[0m"
if [[ $install_rabbitmq == true ]]; then
    # check if RabbitMQ already installed (and fetch version)
    if [ $(command -v 'rabbitmqctl') ]; then
        rabbitmqV=$(sudo rabbitmqctl eval "rabbit_misc:version().")
        log "Existing installation of RabbitMQ found (version $rabbitmqV)"

        # check port
        rabbitmqPort=$(sudo cat /etc/rabbitmq/rabbitmq-env.conf | grep "^.*NODE_PORT\s*=.*" | sed "s/.*NODE\_PORT.*=//g")
        if [[ $rabbitmqPort == \#* || "${#rabbitmqPort}" -eq 0 ]]; then
            # commented port; set to default
            rabbitmqPort=$DEFAULT_PORT_RABBITMQ
        fi

        if [[ $rabbitmqPort != ${rabbitmqCred[4]} ]]; then
            warn "Existing RabbitMQ server is configured to listen to different port than specified ($rabbitmqPort != ${rabbitmqCred[4]}).\nInstallation will continue with currently set port $rabbitmqPort."
            rabbitmqCred[4]=$rabbitmqPort
            saveConfigParam "AIController" "broker_URL" $(assembleURL false ${rabbitmqCred[@]})
        fi

    elif ! $test_only ; then
        # install and configure RabbitMQ
        log "Installing RabbitMQ..."
        sudo apt-get install -y rabbitmq-server | tee -a $logFile

        log "Setting port..."
        sudo sed -i "s/^\s*#\s*NODE_PORT\s*=.*/NODE_PORT=${rabbitmqCred[4]}/g" /etc/rabbitmq/rabbitmq-env.conf

        sudo systemctl enable rabbitmq-server.service
        sudo service rabbitmq-server start
    fi

    if ! $test_only ; then
        # add RabbitMQ user
        log "Creating RabbitMQ user..."
        sudo rabbitmqctl add_user ${rabbitmqCred[1]} ${rabbitmqCred[2]}

        # add new virtual host
        log "Adding virtual host..."
        sudo rabbitmqctl add_vhost ${rabbitmqCred[5]}

        # set permissions
        log "Setting permissions..."
        sudo rabbitmqctl set_permissions -p ${rabbitmqCred[5]} ${rabbitmqCred[1]} ".*" ".*" ".*"

        # restart
        log "Restarting service..."
        sudo service rabbitmq-server stop
        sudo service rabbitmq-server start
        log "Done."
    fi
else
    log "Skipping installation..."
fi



# -----------------------------------------------------------------------------
# INSTALL REDIS
# -----------------------------------------------------------------------------

log "\e[1m[07/11] \e[36mRedis...\e[0m"
if [[ $install_redis == true ]]; then

    # check if Redis already installed
    if [ $(command -v 'redis-cli') ]; then
        redisRunning="$(redis-cli ping)"
        
        if [[ $redisRunning == 'PONG' ]]; then
            log "Existing installation of Redis found"
        fi

        # check port
        redisPort=$(sudo cat /etc/redis/redis.conf | grep "^\s*port\s*.*" | tr -dc '0-9')
        if [[ $redisPort != ${redisCred[4]} ]]; then
            warn "Existing Redis server is configured to listen to different port than specified ($redisPort != ${redisCred[4]}).\nInstallation will continue with currently set port $redisPort."
            redisCred[4]=$redisPort;
            saveConfigParam "AIController" "result_backend" $(assembleURL false ${redisCred[@]})
        fi

    elif ! $test_only ; then
        # install and configure Redis
        log "Installing Redis..."
        sudo apt-get install -y redis-server | tee -a $logFile
        
        # make sure Redis stores its messages in an accessible folder (we're using /var/lib/redis/aide.rdb here)
        log "Setting storage location..."
        sudo sed -i "s/^\s*dir\s*.*/dir \/var\/lib\/redis/g" /etc/redis/redis.conf
        sudo sed -i "s/^\s*dbfilename\s*.*/dbfilename aide.rdb/g" /etc/redis/redis.conf
        
        # also tell systemd
        sudo mkdir -p /etc/systemd/system/redis.service.d
        echo -e "[Service]\nReadWriteDirectories=-/var/lib/redis" | sudo tee -a /etc/systemd/system/redis.service.d/override.conf > /dev/null

        sudo mkdir -p /var/lib/redis
        sudo chown -R redis:redis /var/lib/redis

        # disable persistence. In general, we don't need Redis to save snapshots as it is only used as a result
        # (and therefore message) backend.
        log "Disabling persistence of results..."
        sudo sed -i "s/^\s*save/# save /g" /etc/redis/redis.conf

        # set port
        log "Setting port..."
        sudo sed -i "s/^\s*port\s*.*/port ${redisCred[4]}/g" /etc/redis/redis.conf

        # restart
        log "Restarting service..."
        sudo systemctl daemon-reload
        sudo systemctl enable redis-server.service
        sudo systemctl restart redis-server.service
        log "Done."
    fi
else
    log "Skipping installation..."
fi


# -----------------------------------------------------------------------------
# SETTING UP FILESERVER
# -----------------------------------------------------------------------------

log "\e[1m[08/11] \e[36mFile server...\e[0m"
if [[ $install_fileserver == true && $test_only == false ]]; then
    log "Creating file server directory..."
    sudo mkdir -p $fsDir;
    sudo chown -R $USER $fsDir;
    log "Done."
else
    log "Skipping..."
fi


# -----------------------------------------------------------------------------
# MISC.
# -----------------------------------------------------------------------------

log "\e[1m[09/11] \e[36mAdjusting settings...\e[0m"

# TCP/IP keepalive
if [[ $tcp_keepalive == true && $test_only == false ]]; then
    log "Setting TCP keepalive timer"
    if ! sudo grep -q ^net.ipv4.tcp_keepalive_* /etc/sysctl.conf ; then
        echo "net.ipv4.tcp_keepalive_time = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
        echo "net.ipv4.tcp_keepalive_intvl = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
        echo "net.ipv4.tcp_keepalive_probes = 20" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    else
        sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_time.*/net.ipv4.tcp_keepalive_time = 60 /g" /etc/sysctl.conf
        sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_intvl.*/net.ipv4.tcp_keepalive_intvl = 60 /g" /etc/sysctl.conf
        sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_probes.*/net.ipv4.tcp_keepalive_probes = 20 /g" /etc/sysctl.conf
    fi
    sudo sysctl -p
fi

# environment variables to user profile
if [[ $append_environment_args == true && $test_only == false ]]; then
    log "Appending environment variables to user profile..."
    if ! cat ~/.profile | grep "^export AIDE_CONFIG_PATH=.*" ; then
        echo "export AIDE_CONFIG_PATH=$config_file_out" | tee ~/.profile
    else
        sed -i "s/export AIDE_CONFIG_PATH=.*/export AIDE_CONFIG_PATH=$config_file_out/g" ~/.profile
        log "Updated existing environment variable: 'export AIDE_CONFIG_PATH=$config_file_out'."
    fi
    if ! cat ~/.profile | grep "^export AIDE_MODULES=.*" ; then
        echo "export AIDE_MODULES=$aide_modules" | tee ~/.profile
    else
        sed -i "s/export AIDE_MODULES=.*/export AIDE_MODULES=$aide_modules/g" ~/.profile
        log "Updated existing environment variable: 'export AIDE_MODULES=$AIDE_MODULES'."
    fi
fi



# -----------------------------------------------------------------------------
# TEST INSTALLATION
# -----------------------------------------------------------------------------

log "\e[1m[10/11] \e[36mTesting installation...\e[0m"

# Python
log "Python..." "FALSE" "TRUE"
TEST_python=$($python_exec <<EOF
LIBS= (
    'bottle',
    'gunicorn',
    'psycopg2',
    'tqdm',
    'bcrypt',
    'netifaces',
    'PIL',
    'numpy',
    'requests',
    'celery',
    'cv2',
    'torch',
    'detectron2'
)
import importlib
for lib in LIBS:
    try:
        importlib.import_module(lib)
    except:
        print(lib)
EOF
)
if [ ${#TEST_python} -eq 0 ]; then
    log "\e[32m[ OK ]\e[0m"
else
    log "\e[31m[FAIL]\e[0m"
    log "\tThe following libraries could not be imported: $TEST_python"
fi

# Postgres
log "PostgreSQL..." "FALSE" "TRUE"
TEST_postgres=$(sudo -u postgres PGPASSWORD=$dbPassword psql -w -h $dbHost -p $dbPort -U $dbUser -d $dbName -c 'SELECT name FROM "aide_admin".user;' | grep "$adminName")
if [ ${#TEST_postgres} -gt 0 ]; then
    log "\e[32m[ OK ]\e[0m"
else
    log "\e[31m[FAIL]\e[0m"
    log "\tCheck that the PostgreSQL server is running on the target machine (postgres://$dbUser:****@$dbHost:$dbPort/$dbName), that it is reachable via the respective port and that user $adminName has access to it and the database '$dbName'."
fi

# Python DB access
log "Python database access..." "FALSE" "TRUE"
TEST_python_db=$($python_exec <<EOF
import sys
from util.configDef import Config
from modules.Database.app import Database
try:
    config = Config()
    db = Database(config)
    res = db.execute('SELECT name FROM "aide_admin".user WHERE name=%s;', ('$adminName',), 1)
    if res is not None and len(res) and res[0]['name'] == '$adminName':
        sys.exit(0)
    else:
        raise Exception(f'Invalid response from database "str(res)".')
except Exception as e:
    print(str(e))
EOF
)
if [ ${#TEST_python_db} -eq 0 ]; then
    log "\e[32m[ OK ]\e[0m"
else
    log "\e[31m[FAIL]\e[0m"
    log "\tMessage: '$TEST_python_db'"
fi

# RabbitMQ
# if [[ $install_rabbitmq == true ]]; then
#     #TODO
# fi

# Redis
if [[ $install_redis == true ]]; then
    log "Redis..." "FALSE" "TRUE"
    TEST_redis=$(redis-cli -u $(assembleURL false ${redisCred[@]}) ping);
    if [ "$TEST_redis" == 'PONG' ]; then
        log "\e[32m[ OK ]\e[0m"
    else
        log "\e[31m[FAIL]\e[0m"
        log "\tCheck that Redis is running on the target machine ($(assembleURL true ${redisCred[@]}) and that it is reachable via the respective port."
    fi
fi

# GPU access
if [[ $with_gpu == true ]]; then
    log "GPU access from Python..." "FALSE" "TRUE"
    TEST_python_gpu=$(testPythonGPUaccess)
    if [ ${#TEST_python_gpu} -eq 0 ]; then
        log "\e[32m[ OK ]\e[0m"
    else
        log "\e[31m[FAIL]\e[0m"
        log "\tMessage: '$TEST_python_gpu'"
    fi
fi

# installed AI models
log "Installed AI models..." "FALSE" "TRUE"
    TEST_aic=$($python_exec <<EOF
from util.configDef import Config
from modules.Database.app import Database
from modules.AIController.backend.middleware import AIMiddleware
try:
    config = Config()
    db = Database(config)
    aim = AIMiddleware(config, db, None, True)
    nMod_pred = len(aim.aiModels['prediction'])
    nMod_rank = len(aim.aiModels['ranking'])
    print(f'0 {str(nMod_pred)} {str(nMod_rank)}')
except Exception as e:
    print(f'1 0 0 {str(e)}')
EOF
)
IFS=' ' read -r -a result_aic <<< $TEST_aic
if [ "${result_aic[0]}" = 0 ]; then
    log "\e[32m[ OK ]\e[0m"
    log "\tNumber of prediction models found:   ${result_aic[1]}"
    log "\tNumber of ranking models found:      ${result_aic[2]}"
else
    log "\e[31m[FAIL]\e[0m"
    msg="${result_aic[3]}"
    if [ ${#msg} -gt 0 ]; then
        log "\tMessage: '$msg'"
    fi
fi

# AIController
if [[ $install_aicontroller == false ]]; then
    # only contact FileServer if remote unit
    log "AIController..." "FALSE" "TRUE"
    aicVer="$(wget $aiController_uri)" >> /dev/null
    if [ ${#aicVer} -eq 0 ]; then
        log "\e[31m[FAIL]\e[0m"
        log "\tMake sure remote AIController is running and accessible (check ports, too)."
    fi
fi

# AIWorker  (TODO: could be improved...)
if [[ $install_aiworker == false ]]; then
    # only contact FileServer if remote unit
    log "AIWorker..." "FALSE" "TRUE"
    aiwDetails=$(testAIWorkerAccess)
    if [[ $aiwDetails == "FAIL"* ]]; then
        log "\e[31m[FAIL]\e[0m"
        log "\tMessage: '$aiwDetails'"
    else
        log "\e[32m[ OK ]\e[0m"
        log "\tAIWorker node(s) found:\n$aiwDetails"
    fi
fi

# FileServer
if [[ $install_fileserver == false ]]; then
    # only contact FileServer if remote unit
    log "FileServer..." "FALSE" "TRUE"
    fsVer="$(wget $dataServer_uri)" >> /dev/null
    if [ ${#fsVer} -eq 0 ]; then
        log "\e[31m[FAIL]\e[0m"
        log "\tMake sure remote FileServer is running and accessible (check ports, too)."
    fi
fi


# -----------------------------------------------------------------------------
# SYSTEM PROCESS
# -----------------------------------------------------------------------------

log "\e[1m[11/11] \e[36mSystemd processes...\e[0m"
if [[ $test_only == false && $yes == false && ${#install_daemon} -eq 0 && ( $install_labelUI == true || $install_aicontroller == true ) ]]; then
    # prompt
    log "\nWould you like to install a systemd service for AIDE to start it with the operating system?"
    while true; do
        read -p "[yes/no]" yn
        case $yn in
            [Yy]* ) log "[yes/no] $yn" && install_daemon=true && break;;
            [Nn]* ) log "[yes/no] $yn" && install_daemon=false && break;;
        esac
    done
fi
install_daemon=$(getBool $install_daemon)
if [[ $test_only == false && $install_daemon == true && ( $install_labelUI == true || $install_aicontroller == true ) ]]; then

    # AIDE service group
    if [ $(getent group $aide_group) ]; then
        log "Group '$aide_group' found."
    else
        sudo groupadd "$aide_group"
        sudo chgrp -R $aide_group $aide_root
        sudo chmod g-wx $config_file_out
        sudo chmod g+r $config_file_out
        log "Added group '$aide_group' for AIDE services and updated permissions and ownership of AIDE config file ('$config_file_out')."
    fi

    # config file permissions
    sudo chown $USER:$aide_group $config_file_out
    sudo chmod g+r $config_file_out
    
    if id "$aide_daemon_user" &>/dev/null; then
        # user for daemon script already exists; issue warning
        warn "Specified user for AIDE daemon processes ('$aide_daemon_user') already exists.\nAIDE environment variables will be appended to file '$aide_daemon_user/.profile'."
        homedir=$( getent passwd "$aide_daemon_user" | cut -d: -f6 )
    else
        sudo useradd --shell /bin/bash $aide_daemon_user >> /dev/null
        sudo passwd -l $aide_daemon_user >> /dev/null
        homedir=$( getent passwd "$aide_daemon_user" | cut -d: -f6 )
        sudo mkdir -p $homedir
        sudo chown -R $aide_daemon_user:$aide_daemon_user $homedir
        log "Created user account for AIDE daemon services with name '$aide_daemon_user'."
    fi
    sudo usermod -aG $aide_group $aide_daemon_user
    echo -e "\n# created by AIDE installer ('install_debian.sh') on $(date)." | sudo tee -a $homedir/.profile >> /dev/null;
    echo "export AIDE_CONFIG_PATH=$aide_root" | sudo tee -a $homedir/.profile >> /dev/null;
    echo "export AIDE_MODULES=$aide_modules" | sudo tee -a $homedir/.profile >> /dev/null;
    echo "export PYTHONPATH=$aide_root" | sudo tee -a $homedir/.profile >> /dev/null;
    sudo chown $aide_daemon_user:$aide_daemon_user $homedir/.profile;

    if [[ $install_labelUI == true || $install_aicontroller == true ]]; then
        # Web server daemon
        servicePath="/etc/systemd/system/$SYSTEMD_TARGET_SERVER.service"
        if [ -f "$servicePath" ]; then
            warn "System service for AIDE Web server ('$servicePath') already exists; skipping..."

        else
            gunicorn_exec="$(command -v gunicorn)"
            num_workers=$(getConfigParam "Server" "numWorkers");
            serviceContents=$(cat <<EOF
[Unit]
Description=AIDE Web server
Requires=gunicorn.socket
After=network.target
StartLimitIntervalSec=0

[Service]
Type=notify
Restart=always
RestartSec=1
User=$aide_daemon_user
Group=$aide_group
WorkingDirectory=$aide_root
Environment=AIDE_CONFIG_PATH=$config_file_out
Environment=AIDE_MODULES=$aide_modules
Environment=PYTHONPATH=$aide_root
ExecStart=$gunicorn_exec application:app --bind $serverHost:$serverPort --workers $num_workers
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF
)
            echo -e "$serviceContents" | sudo tee $servicePath >> /dev/null

            sudo systemctl daemon-reload
            sudo systemctl enable $SYSTEMD_TARGET_SERVER.service
            sudo systemctl restart $SYSTEMD_TARGET_SERVER.service
        fi
    else
        log "Skipping installation of AIDE Web frontend service..."
    fi

    if [[ $install_aiworker == true ]]; then
        # Celery services
        tempfile_path=$TMPFILES_VOLATILE_DIR/celery_aide.conf
        servicePath="/etc/systemd/system/$SYSTEMD_TARGET_WORKER.service"
        servicePath_celerybeat="/etc/systemd/system/$SYSTEMD_TARGET_WORKER_BEAT.service"

        # temp file creation
        if [ -f "$tempfile_path" ]; then
            log "Temporary files creation specifier ('$tempfile_path) found."
        else
            sudo mkdir -p "/var/run/celery/"
            sudo chown -R $aide_daemon_user:$aide_group "/var/run/celery"
            sudo mkdir -p "/var/log/celery/"
            sudo chown -R $aide_daemon_user:$aide_group "/var/log/celery"
            sudo mkdir -p $TMPFILES_VOLATILE_DIR
            echo "d /var/run/celery 0755 $aide_daemon_user $aide_group -" | sudo tee -a tempfile_path
            echo "d /var/log/celery 0755 $aide_daemon_user $aide_group -" | sudo tee -a tempfile_path
        fi
        if [[ ${#celery_exec} -eq 0 ]]; then
            celery_exec="$(command -v celery)"
        fi
        tempDir=$(getConfigParam "FileServer" "tempfiles_dir");
        if [ ${#tempDir} -eq 0 ]; then
            tempDir="/tmp/aide/fileserver"
        fi
            
        # Celery daemon config file
        if [ -f $SYSTEMD_CONFIG_WORKER ]; then
            warn "Celery daemon config file ('$SYSTEMD_CONFIG_WORKER') found; will not modify it."      #TODO: modify it nonetheless
        else
            configContents=$(cat <<EOF
CELERYD_NODES="aide@%h"
CELERY_BIN="$celery_exec"
CELERY_APP="celery_worker"
CELERYD_CHDIR="$aide_root"
CELERYD_USER="$aide_daemon_user"
CELERYD_GROUP="$aide_group"
CELERYD_LOG_LEVEL="INFO"
CELERYD_PID_FILE="$tempDir/celeryd_aide.pid"
CELERYD_LOG_FILE="/var/log/celery/celeryd_aide.log"
CELERYD_OPTS=""
CELERY_CREATE_DIRS=1
CELERYBEAT_CHDIR="$aide_root"
CELERYBEAT_OPTS="-s $tempDir"

# AIDE environment variables
export AIDE_CONFIG_PATH=$config_file_out
export AIDE_MODULES=$aide_modules
export PYTHONPATH=$aide_root
EOF
)
            echo -e "$configContents" | sudo tee $SYSTEMD_CONFIG_WORKER >> /dev/null
        fi

        # AIWorker daemon
        if [ -f "$servicePath" ]; then
            warn "System service for AIWorker ('$SYSTEMD_TARGET_WORKER') already exists; skipping..."
        else
            serviceContents=$(cat <<EOF
[Unit]
Description=Celery Service for AIDE AIWorker
After=network.target
After=rabbitmq-server.service
After=redis.service

[Service]
Type=forking
User=$aide_daemon_user
Group=$aide_group
EnvironmentFile=$SYSTEMD_CONFIG_WORKER
WorkingDirectory=$aide_root
ExecStart=$python_exec \${CELERY_BIN} -A \$CELERY_APP multi start \$CELERYD_NODES \
    --pidfile=\${CELERYD_PID_FILE} --logfile=\${CELERYD_LOG_FILE} \
    --loglevel="\${CELERYD_LOG_LEVEL}" \$CELERYD_OPTS
ExecStop=$python_exec \${CELERY_BIN} multi stopwait \$CELERYD_NODES \
    --pidfile=${CELERYD_PID_FILE} --loglevel="\${CELERYD_LOG_LEVEL}"
ExecReload=$python_exec \${CELERY_BIN} -A \$CELERY_APP multi restart \$CELERYD_NODES \
    --pidfile=\${CELERYD_PID_FILE} --logfile=\${CELERYD_LOG_FILE} \
    --loglevel="\${CELERYD_LOG_LEVEL}" \$CELERYD_OPTS
Environment=AIDE_CONFIG_PATH=$config_file_out
Environment=AIDE_MODULES=$aide_modules
Environment=PYTHONPATH=$aide_root
Restart=always

[Install]
WantedBy=multi-user.target
EOF
)
            echo -e "$serviceContents" | sudo tee $servicePath >> /dev/null
            sudo systemctl daemon-reload
            sudo systemctl enable $SYSTEMD_TARGET_WORKER.service
            sudo systemctl restart $SYSTEMD_TARGET_WORKER.service
        fi

        # AIWorker daemon celerybeat
        if [ -f "$servicePath_celerybeat" ]; then
            warn "System service for AIWorker periodic checking ('$SYSTEMD_TARGET_WORKER_BEAT') already exists; skipping..."
        else
            # celerybeat script
            serviceContents_celerybeat=$(cat <<EOF
[Unit]
Description=Celery Beat Service for AIDE AIWorker
After=network.target
After=rabbitmq-server.service
After=redis.service

[Service]
Type=simple
User=$aide_daemon_user
Group=$aide_group
EnvironmentFile=$SYSTEMD_CONFIG_WORKER
WorkingDirectory=$aide_root
ExecStart=/bin/sh -c '${CELERY_BIN} -A ${CELERY_APP} beat  \
    --pidfile=${CELERYBEAT_PID_FILE} \
    --logfile=${CELERYBEAT_LOG_FILE} --loglevel=${CELERYD_LOG_LEVEL}'
Restart=always

[Install]
WantedBy=multi-user.target     
EOF
)
            echo -e "$serviceContents_celerybeat" | sudo tee $servicePath_celerybeat >> /dev/null
            sudo systemctl daemon-reload
            sudo systemctl enable $SYSTEMD_TARGET_WORKER_BEAT.service
            sudo systemctl restart $SYSTEMD_TARGET_WORKER_BEAT.service
        fi
    else
        log "Skipping installation of AIWorker daemon..."
    fi
        
    log "System processes set up. You can start/stop/restart them with the commands as below:"
    log "\tAIDE Web server: 'sudo service $SYSTEMD_TARGET_SERVER [start|stop|restart]'"
    if [[ $install_aiworker == true ]]; then
        log "\tAIDE AIWorker:   'sudo service $SYSTEMD_TARGET_WORKER [start|stop|restart]'"
    fi
    log "\nTo view logs about the AIDE services:"
    log "\tAIDE Web server:                 'sudo journalctl -u $SYSTEMD_TARGET_SERVER'"
    if [[ $install_aiworker == true ]]; then
        log "\tAIDE AIWorker:                   'sudo journalctl -u $SYSTEMD_TARGET_WORKER'"
        log "\tAIDE AIWorker periodic checker:  'sudo journalctl -u $SYSTEMD_TARGET_WORKER_BEAT'"
    fi
    log "\nIf everything went correctly AIDE should now be running and reachable in your Web browser:"
    log "\thttp://$HOSTNAME:$serverPort"
    log "\nYou can now log in with your administrator account '$adminName'."

else
    log "Skipping..."
fi



# -----------------------------------------------------------------------------
# FINALIZE
# -----------------------------------------------------------------------------

log "\nInstallation of AIDE completed.\nLog written to file '$logFile'."
