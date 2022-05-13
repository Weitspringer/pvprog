#!/bin/bash

# Create a new namespace and run the program
myct_isolation::in_new_namespace() {
    local -r rootfs="$1" program="$2" args="$3"q
    unshare --mount --pid --fork --mount-proc=$rootfs/proc chroot $rootfs
}

# Run program in another existing namespace
myct_isolation::in_entered_namespace() {
    local -r rootfs="$1" kind="$2" pid="$3" program="$4" args="$5"
    echo $kind
    case $kind in
        mount) nsenter --target $pid -m unshare --fork --root=$rootfs --mount-proc;;
        uts) nsenter --target $pid -u unshare --fork --root=$rootfs --mount-proc ;;
        ipc) nsenter --target $pid -i unshare --fork --root=$rootfs --mount-proc ;;
        network) nsenter --target $pid -n unshare --fork --root=$rootfs --mount-proc ;;
        pid) nsenter --target $pid -p unshare --fork --root=$rootfs --mount-proc ;;
        cgroup) nsenter --target $pid -C unshare --fork --root=$rootfs --mount-proc ;;
        user) nsenter --target $pid -U unshare --fork --root=$rootfs --mount-proc ;;
        time) nsenter --target $pid -T unshare --fork --root=$rootfs --mount-proc ;;
        default) nsenter --target $pid -m -p unshare --fork --root=$rootfs --mount-proc ;;
        *) 
            echo "Unknown namespace type: $kind." 
            echo "Possible types: mount, uts, ipc, network, pid, cgroup, user, time, and default."
            echo "I'll put you into a new namespace instead."
            myct_isolation::in_new_namespace $rootfs
            ;;
    esac
}
