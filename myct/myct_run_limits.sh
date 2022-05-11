#!/bin/bash


myct_run_limits::_get_cgroup_path(){
    local -r controller = $1
    return "/sys/fs/cgroup/$controller/myct"
}


myct_run_limits::_mount_cgroup_fs(){
    # systemd(1) may automatically mounts the cgroup2 filesystem 
    # at /sys/fs/cgroup/unified during the boot process
    if [! -d "/sys/fs/cgroup/unified"];
    then
        mount -t cgroup2 none /sys/fs/cgroup/unified
    fi
}


myct_run_limits::_unmount_cgroup_fs(){
    # first remove all child cgroups, which in turn can be done only after 
    # all member processes have been moved from those cgroups to the root cgroup
    # then unmount, eg.g.: umount /sys/fs/cgroup/pids

}


myct_run_limits::_create_cgroup(){
    local -r controller = $1
    local -r path = myct_run_limits::_get_cgroup_path $controller

    if [! -d "$path"];
    then
        mkdir -p "$path"
    fi
}


myct_run_limits::_remove_cgroup(){
    # it must first have no child cgroups and contain no (nonzombie) processes
    # Question -> how to check for child groups / for processes
    #   (1) child groups -> subfolders
    #   (2) processes -> cgroup.proc file (?)
    local -r controller = $1
    local -r path = myct_run_limits::_get_cgroup_path $controller

    rm -r "$path"
}


myct_run_limits::_move_process_to_cgroup(){
    local -r pid = $1
    local -r controller = $2
    local -r path = myct_run_limits::_get_cgroup_path $controller

    echo "$pid" >> "$path/cgroup.procs"
}


myct_run_limits::_remove_process_from_cgroup(){
    # not sure ->probably just empty the controller.proc file (?)
    #       - no, emptying would remove all processes
}


myct_run_limits::_add_limit(){
    local -r controller = $1
    local -r key = $2
    local -r value = $3
    local -r path = myct_run_limits::_get_cgroup_path $controller

    echo "$value" > "$path/$controller.$key"

}


myct_run_limits::limit(){
    # recource limitations implemented via cgroups
    # steps:
    #   (1) mount cgroup file system
    #   (2) create cgroup per controller
    #   (3) add the limit

    local -r controller = $1
    local -r key = $2
    local -r value = $3

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

    local -r controller = $1
    local -r pid = $2
    
    myct_run_limits::_mount_cgroup_fs
    myct_run_limits::_create_cgroup $controller
    myct_run_limits::_move_process_to_cgroup $pid $controller
}