#!/usr/bin/env sh
# Download the four MNIST files into training/data/.
set -e
cd "$(dirname "$0")"
mkdir -p data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f "data/$f.gz" ] || curl -sSfL -o "data/$f.gz" "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
echo "MNIST ready in training/data/"
