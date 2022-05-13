#!/bin/bash

# Create a new namespace and run the program
myct_isolation::in_new_namespace() {
    local -r rootfs="$1" program="$2" args="$3"
    echo "let's run the executable: $program"
    unshare --mount --pid --fork --root=$rootfs --mount-proc $program $args &
}

# Run program in another existing namespace
myct_isolation::in_entered_namespace() {
    local -r rootfs="$1" kind="$2" pid="$3" program="$4" args="$5"
    echo $kind
    case $kind in
        mount) nsenter --target $pid -m unshare --fork --root=$rootfs --mount-proc $program $args ;;
        uts) nsenter --target $pid -u unshare --fork --root=$rootfs --mount-proc $program $args ;;
        ipc) nsenter --target $pid -i unshare --fork --root=$rootfs --mount-proc $program $args ;;
        network) nsenter --target $pid -n unshare --fork --root=$rootfs --mount-proc $program $args ;;
        pid) nsenter --target $pid -p unshare --fork --root=$rootfs --mount-proc $program $args ;;
        cgroup) nsenter --target $pid -C unshare --fork --root=$rootfs --mount-proc $program $args ;;
        user) nsenter --target $pid -U unshare --fork --root=$rootfs --mount-proc $program $args ;;
        time) nsenter --target $pid -T unshare --fork --root=$rootfs --mount-proc $program $args ;;
        default) nsenter --target $pid -m -p unshare --fork --root=$rootfs --mount-proc $program $args ;;
        *) 
            echo "Unknown namespace type: $kind." 
            echo "Possible types: mount, uts, ipc, network, pid, cgroup, user, time, and default."
            echo "I'll put you into a new namespace instead."
            myct_isolation::in_new_namespace $rootfs $program $args # run given process in our container
            ;;
    esac
}
