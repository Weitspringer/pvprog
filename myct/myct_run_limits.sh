#!/bin/bash

declare cgroup_version=0


myct_run_limits::set_cgroup_version(){
    local v1=$(mount -l | grep -c cgroup)
    local v2=$(mount -l | grep -c cgroup2)

    echo "anzahl v1: $v1, anzahl v2: $v2"

    # we will go with cgroups v1 only

    if [[ $v1 > "0" ]];
    then
        cgroup_version=1;
    else
        cgroup_version=0;
    fi
}


myct_run_limits::_get_cgroup_path(){
    local -r controller=$1
    echo "/sys/fs/cgroup/$controller/myct"
}


myct_run_limits::_mount_cgroup_fs(){
    # systemd(1) may automatically mounts the cgroup2 filesystem 
    # at /sys/fs/cgroup/unified during the boot process
    if [[ ! $cgroup_version > "0" ]];
    then
        # sudo mkdir -p "/sys/fs/cgroup"
        sudo mount -t cgroup2 none /sys/fs/cgroup
        myct_run_limits::set_cgroup_version
    fi
}


myct_run_limits::_unmount_cgroup_fs(){
    # first remove all child cgroups, which in turn can be done only after 
    # all member processes have been moved from those cgroups to the root cgroup
    # then unmount, e.g.: umount /sys/fs/cgroup/pids
    echo "not implemented yet"

}


myct_run_limits::_create_cgroup(){
    local -r controller=$1
    local -r path=$(myct_run_limits::_get_cgroup_path $controller)

    if [ ! -d "$path" ];
    then
        sudo mkdir -p "$path"
        echo "cgroup created"
    fi
}


myct_run_limits::_remove_cgroup(){
    # it must first have no child cgroups and contain no (nonzombie) processes
    # Question -> how to check for child groups / for processes
    #   (1) child groups -> subfolders
    #   (2) processes -> cgroup.proc file (?)
    local -r controller=$1
    local -r path=$(myct_run_limits::_get_cgroup_path $controller)

    rm -r "$path"
}


myct_run_limits::_move_process_to_cgroup(){
    local -r pid=$1
    local -r controller=$2
    local -r path=$(myct_run_limits::_get_cgroup_path $controller)

    echo "$pid" | sudo tee -a "$path/cgroup.procs"
    cat "$path/cgroup.procs"
}


myct_run_limits::_remove_process_from_cgroup(){
    # not sure ->probably just empty the controller.proc file (?)
    #       - no, emptying would remove all processes
    echo "not implemented yet"
}


myct_run_limits::_add_limit(){
    local -r controller=$1
    local -r key=$2
    local -r value=$3
    local -r path=$(myct_run_limits::_get_cgroup_path $controller)

    sudo touch "$path/$controller.$key"
    echo "$value" | sudo tee "$path/$controller.$key"

}


myct_run_limits::limit(){
    # recource limitations implemented via cgroups
    # steps:
    #   (1) mount cgroup file system
    #   (2) create cgroup per controller
    #   (3) add the limit

    local -r controller=$1
    local -r key=$2
    local -r value=$3

    myct_run_limits::set_cgroup_version
    myct_run_limits::_mount_cgroup_fs
    myct_run_limits::_create_cgroup $controller
    myct_run_limits::_add_limit $controller $key $value
}


myct_run_limits::add_process(){
    # recource limitations implemented via cgroups
    # steps:
    #   (1) mount cgroup file system
    #   (2) create cgroup per controller
    #   (3) move process to cgroup

    local -r controller=$1
    local -r pid=$2
    
    myct_run_limits::set_cgroup_version
    myct_run_limits::_mount_cgroup_fs
    myct_run_limits::_create_cgroup $controller
    myct_run_limits::_move_process_to_cgroup $pid $controller
}