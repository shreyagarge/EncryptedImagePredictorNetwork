#! /bin/bash
base_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

xhost

ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')

xhost + $ip

echo $ip

docker run -e DISPLAY=$ip:0 -v $base_path/Predictor:/SEAL/Predictor -it seal python3 "$@"