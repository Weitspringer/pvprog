#!/bin/bash

source "./myct_run_limits.sh"
source "myct_isolation.sh"

declare -r SCRIPT_MODE="$1"
# declare -r CONTAINER_PATH=$(readlink -f $2)
declare -r CONTAINER_PATH="$2"


echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

echo "$@"

# Script has to be executed with admin privileges

case $SCRIPT_MODE in
    init)
        # Setup a root file system using debootstrap
        # myct init <container-path>
        # Check requirements
        if [ $# = 2 ]; 
        then            
            REQUIRED_PKG="debootstrap"
            PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
            echo Checking for $REQUIRED_PKG
            if [ "" = "$PKG_OK" ]; 
            then
                echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
                sudo apt-get -y install $REQUIRED_PKG
            else
                echo "$REQUIRED_PKG already installed. Proceeding..."
            fi

            # Check if container path exists
            if [ ! -d $CONTAINER_PATH ]
            then
                echo "$CONTAINER_PATH not found. Setting up $CONTAINER_PATH."
                mkdir -p "$CONTAINER_PATH"
            else
                echo "$CONTAINER_PATH found. Proceeding..."
            fi

            # Read container path argument and install Debian root file system
            echo "Installing container in: $CONTAINER_PATH..."
            sudo debootstrap stable $CONTAINER_PATH http://deb.debian.org/debian
        else
            echo "Wrong number of arguments. Usage of init: myct init <container-path>"
        fi
        ;;
    map)
        declare -r HOST_PATH="$3"
        declare -r TARGET_PATH="$2$4"

        echo "map host folder $HOST_PATH to $TARGET_PATH"
        # Map host directories read-only into container
        # myct map <container-path> <host-path> <target-path>
        [ ! -d $TARGET_PATH ] && mkdir -p "$TARGET_PATH"
        mount -r --bind "$HOST_PATH" "$TARGET_PATH"
        ;;
    unmap)
        declare -r TARGET_PATH="$2$3"

        echo "unmap host folder $HOST_PATH to $TARGET_PATH"
        [ ! -d $TARGET_PATH ] && mkdir -p "$TARGET_PATH"
        umount "$TARGET_PATH"
        ;;
    run)
        # Transform long options to short ones
        for arg in "$@"; do
            shift
            case "$arg" in
                "--namespace") set -- "$@" "-n" ;;
                "--limit") set -- "$@" "-l" ;;
                *)        set -- "$@" "$arg"
            esac
        done

        optstring="n:l:"
        echo "${optstring}"

        OPTIND=3    # set index 2 since the first parameter is then mode, in this case "run"
        while getopts ${optstring} arg; do
            case ${arg} in
                n) 
                    echo "Namespace: ${OPTARG}"
                    namespaceArray=(${OPTARG//=/ }) # split string in array by delimiter: '='
                    KIND="${namespaceArray[0]}"
                    PID="${namespaceArray[1]}" 
                    ;;
                l) 
                    echo "Limit: ${OPTARG}" 
                    limitArray=(${OPTARG//=/ }) # split string in array by delimiter: '='
                    CONTROLLER_KEY="${limitArray[0]}"
                    VALUE="${limitArray[1]}"
                    limitArray=(${CONTROLLER_KEY//./ }) # split string in array by delimiter: '.'
                    CONTROLLER="${limitArray[0]}"
                    KEY="${limitArray[1]}"
                    ;;
                ?) echo "Unkown option: -${OPTARG}" ;;
            esac
        done
        shift $(($OPTIND - 1))
        printf "Remaining arguments are: %s\n$*\n"
        EXECUTABLE=$CONTAINER_PATH$1
        # If --namespace option is given, the user wants the process running in the container to join the namespace of another process in a container
        # Check: If no --namespace option is given, create new namespace with 'unshare --fork --root="$CONTAINER_PATH" --mount-proc="$CONTAINER_PATH/proc"'
        if [ -v KIND ] && [ -v PID ];
        then
            echo "enter namespace: $KIND for process: $PID"
            myct_isolation::in_entered_namespace $CONTAINER_PATH $KIND $PID "/bin/bash";
        else
            echo "new namespace"
            myct_isolation::in_new_namespace $CONTAINER_PATH "/bin/bash";
        fi

        # apply resource management
        myct_run_limits::limit $CONTROLLER $KEY $VALUE
        myct_run_limits::add_process $CONTROLLER $$

        # run given process in our container
        echo "let's run the executable: $EXECUTABLE"
        exec $EXECUTABLE &
        ;;
    *)
        echo "Unknown command $SCRIPT_MODE"
        ;;
esac

#if [ $1 = "init" ]
#then
#    echo "init"
#elif [ $1 = "map" ]
#then
#    echo "map"       
#elif [ $1 = "run"]
#then
#    echo "run"
#else
#    echo "Hallo Welt"
#fi 

# Setup a root file system using debootstrap
# myct init <container-path>
# Download debootstrap if not already existing
#apt-get install debootstrap
# Create container root directory
#mkdir <container-path>
# Download debian distribution via debootstrap
#debootstrap stable <container-path> http://deb.debian.org/debian

# Map host directories read-only into container
# myct map <container-path> <host-path> <target-path>
#mkdir <container-path>/<target-path>
#mount -r --bind <host-path> <container-path>/<target-path>

# Run file executable in container with passed arguments
# myct run <container-path> [options] <executable> [args...]
# Options: --namespace <kind>=<pid> --limit <controller.key>=<value>

# Create minimal kernel namespace and limit filesystem access
#unshare -p -f chroot <container-path>
# Mount
#mount -t proc proc /proc


# some links regarding bash
# https://www.cyberciti.biz/faq/bash-check-if-file-does-not-exist-linux-unix/
# https://koenwoortman.com/bash-script-constants/
# https://www.man7.org/linux/man-pages/man1/getopts.1p.html
