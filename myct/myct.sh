#!/bin/bash
echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

# Script has to be executed with admin privileges

case $1 in
    init)
        # Setup a root file system using debootstrap
        # myct init <container-path>
        # Check requirements
        if [ $# = 2 ]; then
            REQUIRED_PKG="debootstrap"
            PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
            echo Checking for $REQUIRED_PKG
            if [ "" = "$PKG_OK" ]; then
                echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
                sudo apt-get -y install $REQUIRED_PKG
                else
                echo "$REQUIRED_PKG already installed. Proceeding..."
            fi
            # Read container path argument and install Debian root file system
            echo "Installing container in $(cd "$(dirname "$2")"; pwd)/$(basename "$2")"
            [ -f $2 ] && mkdir $2
            sudo debootstrap stable $2 http://deb.debian.org/debian
            else
            echo "Wrong number of arguments. Usage of init: myct init <container-path>"
        fi
        ;;
    map)
        echo map
        ;;
    run)
        echo run
        ;;
    *)
        echo Unknown command "$1"
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