# Emulation

To conduct experiment on a GPU cluster, we provide docker container scripts to install required Torch, CUDA, and NCCL packages.

## Container
Build with Docker locally
```
docker build --network=host -f emulation-env/Dockerfile -t <user>/opus:latest .
```

Push to Docker Hub
```
docker login
docker push <user>/opus:latest
```

Pull inside target environment:

Note: we run experiments on Perlmutter supercomputer, so we use `shifterimg/shifter`, replacing `docker`

```
shifterimg login docker.io
shifterimg pull <user>/opus:latest
```
On other GPU clusters:
```
docker login docker.io
docker pull <user>/opus:latest
```
