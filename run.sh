docker pull ib16/mnist-transfer-learning:latest
docker run -t --rm -v volume-mnist:/app/state mnist-transfer-learning --fit
docker run -t --rm -v volume-mnist:/app/state mnist-transfer-learning --transfer --epochs=100 --early-stopping=25