import argparse

from kubeflow.training import Trainer, TrainingClient


def train_fashion_mnist(dict):

    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.nn.parallel import DistributedDataParallel
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision.datasets import FashionMNIST

    backend = dict.get("backend")
    batch_size = dict.get("batch_size")
    test_batch_size = dict.get("test_batch_size")
    epochs = dict.get("epochs")
    lr = dict.get("lr")
    lr_gamma = dict.get("lr_gamma")
    lr_period = dict.get("lr_period")
    seed = dict.get("seed")
    log_interval = dict.get("log_interval")
    save_model = dict.get("save_model")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, 10),
                nn.ReLU(),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    def train(model, device, criterion, train_loader, optimizer, epoch, log_interval):
        # Enter training mode
        model.train()
        # Iterate over mini-batches from the training set
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Copy the data to the GPU device if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def evaluate(model, device, criterion, rank, test_loader, epoch):
        # Enter evaluation mode
        model.eval()
        samples = 0
        local_loss = 0
        local_correct = 0
        # Disable gradient computation to speed up the computation
        # and reduce memory usage
        with torch.no_grad():
            # Iterate over mini-batches from the evaluation set
            for inputs, labels in test_loader:
                samples += len(inputs)
                # Copy the data to the GPU device if available
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Sum up batch loss
                local_loss += criterion(outputs, labels).item()
                # Get the index of the max log-probability
                pred = outputs.argmax(dim=1, keepdim=True)
                local_correct += pred.eq(labels.view_as(pred)).sum().item()

        local_accuracy = 100.0 * local_correct / samples

        header = f"{'-'*15} Epoch {epoch} Evaluation {'-'*15}"
        print(f"\n{header}\n")
        # Log local metrics on each rank
        print(f"Local rank {rank}:")
        print(f"- Loss: {local_loss / samples:.4f}")
        print(f"- Accuracy: {local_correct}/{samples} ({local_accuracy:.0f}%)\n")

        # To Tensors so local metrics can be globally reduced across ranks
        global_loss = torch.tensor([local_loss], device=device)
        global_correct = torch.tensor([local_correct], device=device)

        # Reduce the metrics on rank 0
        dist.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        dist.reduce(global_correct, dst=0, op=torch.distributed.ReduceOp.SUM)

        # Log the aggregated metrics only on rank 0
        if rank == 0:
            global_loss = global_loss / len(test_loader.dataset)
            global_accuracy = (global_correct.double() / len(test_loader.dataset)) * 100
            global_correct = global_correct.int().item()
            samples = len(test_loader.dataset)
            print("Global metrics:")
            print(f"- Loss: {global_loss.item():.6f}")
            print(
                f"- Accuracy: {global_correct}/{samples} ({global_accuracy.item():.2f}%)"
            )
        else:
            print("See rank 0 logs for global metrics")
        print(f"\n{'-'*len(header)}\n")

    dist.init_process_group(backend=backend)

    torch.manual_seed(seed)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    model = Net()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if backend != torch.distributed.Backend.NCCL:
            print(
                "Please use NCCL distributed backend for the best performance using NVIDIA GPUs"
            )
        device = torch.device(f"cuda:{local_rank}")
        model = DistributedDataParallel(model.to(device), device_ids=[local_rank])
    else:
        device = torch.device("cpu")
        model = DistributedDataParallel(model.to(device))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets and data loaders for training & validation, download if necessary
    training_set = FashionMNIST(
        "./data", train=True, transform=transform, download=True
    )
    training_sampler = DistributedSampler(training_set)
    training_loader = DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        sampler=training_sampler,
        pin_memory=use_cuda,
    )

    validation_set = FashionMNIST(
        "./data", train=False, transform=transform, download=True
    )
    validation_sampler = DistributedSampler(validation_set)
    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=test_batch_size,
        sampler=validation_sampler,
        pin_memory=use_cuda,
    )

    # Report dataset sizes
    print("Training set has {} instances".format(len(training_set)))
    print("Validation set has {} instances\n".format(len(validation_set)))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, criterion, training_loader, optimizer, epoch, log_interval)
        evaluate(model, device, criterion, rank, validation_loader, epoch)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist.pt")

    # Wait so rank 0 can gather the global metrics
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch DDP Fashion MNIST Training Example"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training [100]",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing [100]",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train [10]",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        metavar="LR",
        help="learning rate [1e-1]",
    )

    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        metavar="G",
        help="learning rate decay factor [0.5]",
    )

    parser.add_argument(
        "--lr-period",
        type=float,
        default=20,
        metavar="P",
        help="learning rate decay period in step size [20]",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="S",
        help="random seed [0]",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training metrics [10]",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="saving the trained model [False]",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["gloo", "nccl"],
        default="nccl",
        help="Distributed backend [nccl]",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of workers [1]",
    )

    parser.add_argument(
        "--worker-resources",
        type=str,
        nargs=2,
        action="append",
        dest="resources",
        default=[
            ("cpu", 1),
            ("memory", "2Gi"),
            ("nvidia.com/gpu", 1),
        ],
        metavar=("RESOURCE", "QUANTITY"),
        help="Resources per worker [cpu: 1, memory: 2Gi, nvidia.com/gpu: 1]",
    )

    parser.add_argument(
        "--runtime",
        type=str,
        default="torch-distributed",
        metavar="NAME",
        help="the training runtime [torch-distributed]",
    )

    args = parser.parse_args()

    client = TrainingClient()

    job_name = client.train(
        runtime_ref=args.runtime,
        trainer=Trainer(
            func=train_fashion_mnist,
            func_args={
                "backend": args.backend,
                "batch_size": args.batch_size,
                "test_batch_size": args.test_batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_gamma": args.lr_gamma,
                "lr_period": args.lr_period,
                "seed": args.seed,
                "log_interval": args.log_interval,
                "save_model": args.save_model,
            },
            num_nodes=args.num_workers,
            resources_per_node={
                resource: quantity for (resource, quantity) in args.resources
            },
        ),
    )

    client.get_job_logs(job_name, follow=True)
