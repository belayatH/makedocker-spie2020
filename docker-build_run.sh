#!/bin/bash

# Build docker machine
nvidia-docker build -t spie20 .

# start machine
# nvidia-docker run --rm -i -t \

nvidia-docker run -i -t --restart=always \
    spie20 \
    /bin/bash

#-u $(id -u $(whoami)):$(id -g $(whoami)) \    
#-e HOME=${HOME} \
#-v $(pwd):/src -w /src \
#-v /mnt/hpc-home/data/hand-rheumatism:/${HOME}/data/hand-rheumatism \
#-v ${HOME}/data/_out:${HOME}/data/_out \
#-e PYTHONPATH=/src \
#hand-rheumatism \
#bash
