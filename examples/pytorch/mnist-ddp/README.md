# PyTorch DDP Fashion MNIST Training Example

This example demonstrates how to train a neural network to classify images
using the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset
and [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

You can either run this example with the provided Jupyter notebook,
or by running the Python script directly.

In any case, you need to install the Kubeflow training v2 control plane
on your Kubernetes cluster, if it's not already deployed:

```console
kubectl apply --server-side -k "https://github.com/kubeflow/training-operator.git/manifests/v2/overlays/standalone?ref=master"
```

## Jupyter Notebook

You can set up your environment by running the following commands:

```console
python -m venv .venv
source .venv/bin/activate
pip install jupyter
```

And start the notebook by running:

```console
jupyter notebook examples/pytorch/mnist-ddp/mnist.ipynb
```

You can then access the notebook from your Web browser and follow the instructions.

## Python Script

### Setup

You need to set up the Python environment on your local machine or client:

```console
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/kubeflow/training-operator.git@master#subdirectory=sdk_v2
```

You can refer to the [training operator documentation](https://www.kubeflow.org/docs/components/training/installation/)
for more information.

### Usage

```console
python mnist.py --help
usage: mnist.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR] [--lr-gamma G] [--lr-period P] [--seed S] [--log-interval N] [--save-model]
                [--backend {gloo,nccl}] [--num-workers N] [--worker-resources RESOURCE QUANTITY] [--runtime NAME]

PyTorch DDP Fashion MNIST Training Example

options:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training [100]
  --test-batch-size N   input batch size for testing [100]
  --epochs N            number of epochs to train [10]
  --lr LR               learning rate [1e-1]
  --lr-gamma G          learning rate decay factor [0.5]
  --lr-period P         learning rate decay period in step size [20]
  --seed S              random seed [0]
  --log-interval N      how many batches to wait before logging training metrics [10]
  --save-model          saving the trained model [False]
  --backend {gloo,nccl}
                        Distributed backend [nccl]
  --num-workers N       Number of workers [1]
  --worker-resources RESOURCE QUANTITY
                        Resources per worker [cpu: 1, memory: 2Gi, nvidia.com/gpu: 1]
  --runtime NAME        the training runtime [torch-distributed]
```

### Example

Train the model on 8 worker nodes using 1 NVIDIA GPU each:

```console
python mnist.py \
    --num-workers 4 \
    --worker-resources "nvidia.com/gpu" 1 \
    --worker-resource cpu 4 \
    --worker-resources memory 16Gi \
    --epochs 100 \
    --batch-size 100 \
    --lr 1e-1 \
    --lr-period 25 \
    --lr-gamma 0.7
```

At the end of each epoch, local metrics are printed in each worker logs and the global metrics
are gathered and printed in the rank 0 worker logs.

When the training completes, you should see the following at the end of the rank 0 worker logs:

```text
--------------- Epoch 50 Evaluation ---------------

Local rank 0:
- Loss: 0.0039
- Accuracy: 2258/2500 (90%)

Global metrics:
- Loss: 0.004262
- Accuracy: 9023/10000 (90.23%)

---------------------------------------------------
```
