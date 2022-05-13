# Welcome to myct!

This is a minimal container runtime written as a shell script. It is able to setup a root file system using debootstrap, allows existing host directories to be mapped read-only and limit filesystem access. Myct can create a minimal kernel namespace to prevent signals to be send to other processes outside of the container. New processes in the container are able to join existing namespaces and ressource usage like memeory and cpu can be limited.

Firstly, execute the following create a container in any given directoy:

    $ myct init <container-path>

After that, mount a host directory read-only into the container at the given destination:

    $ myct map <container-path> <host-path> <target-path>

 To run a file executable in the container run the following command: 

    $ myct run <container-path> [options] <executable> [args...]

The run command can be executed with options. If "--namespace" is not specified, there will be a new namespace created with the container. If you specify a namespace you have to know the PID of the process already running in the isolated container.
The possible options are:

    --namespace <kind>=<pid>
    --limit <controller.key>=<value>

Proper clean-up and deletion of the container has to be done manually, as this was not part of the excercise. 

The test cases provided only test the limit option.