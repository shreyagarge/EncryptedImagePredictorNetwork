#! /bin/bash
base_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

docker run --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $base_path/Predictor:/SEAL/Predictor -it seal python3 "$@"
