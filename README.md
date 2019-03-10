# Neural Networks for Encrypted Data using Homomorphic Encryption

Python implementation of a Neural Network for recognising encrypted images of handwritten digits (MNIST dataset).
Watch video [here](https://youtu.be/6EapT7HAvFA)

#### Dependencies

* [docker-ce](https://docs.docker.com/install/)

#### To run:

	cd Encrypted_NN

	./build_docker.sh # builds the docker images with all the required libraries

	./run_docker_<os>.sh Predictor/client.py \<image filename (from /Predictor/images folder)> \<debug (optional)>

Example: `./run_docker_<os>.sh Predictor/client.py testim1`



