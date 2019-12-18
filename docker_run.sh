# Build docker machine/ docker image
nvidia-docker build -t spie20 .

# start machine / docker container
# nvidia-docker run --rm -i -t \

nvidia-docker run -i -t --restart=always \
    spie20 \
    /bin/bash

