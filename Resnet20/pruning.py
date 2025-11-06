import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import time
from tqdm import tqdm
import argparse






class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)





def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return running_loss / total, correct / total


def print_sparsity(model):
    total_params, total_zero = 0, 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_mask"):
            mask = module.weight_mask.detach().cpu()
            num_params = mask.numel()
            num_zeros = num_params - mask.sum().item()
            sparsity = 100.0 * num_zeros / num_params
            print(
                f"Layer {name}: {sparsity:.2f}% pruned ({num_params - num_zeros}/{num_params} remain)"
            )
            total_params += num_params
            total_zero += num_zeros
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu()
            num_params = weight.numel()
            num_zeros = (weight == 0).sum().item()
            sparsity = 100.0 * num_zeros / num_params
            print(
                f"Layer {name} (unpruned): {sparsity:.2f}% pruned ({num_params - num_zeros}/{num_params} remain)"
            )
            total_params += num_params
            total_zero += num_zeros
    if total_params > 0:
        total_sparsity = 100.0 * total_zero / total_params
        print(
            f"--> Total Model Sparsity: {total_sparsity:.2f}% pruned ({total_params - total_zero}/{total_params} remain)\n"
        )


def verify_excluded_layers(model):
    """Check that shortcut convs and final fc layer are not pruned."""
    for name, module in model.named_modules():
        if (isinstance(module, nn.Conv2d) and "shortcut" in name) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_mask"):
                print(f"[WARNING] Excluded layer {name} has been pruned!")
            else:
                print(f"[OK] Excluded layer {name} is unpruned as expected.")





def main(args):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    base_train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_size = int(0.9 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        base_train_dataset, [train_size, val_size]
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    
    forgetting_scores = torch.load("forgetting_scores.pt")
    n_train = len(forgetting_scores)
    num_selected = int(args.percent * n_train)
    sorted_indices = torch.argsort(forgetting_scores)

    if args.difficulty == "e":
        coreset_indices = sorted_indices[:num_selected]  
    elif args.difficulty == "h":
        coreset_indices = sorted_indices[-num_selected:]  
    else:
        raise ValueError("difficulty flag must be 'e' or 'h'")

    coreset_dataset = Subset(base_train_dataset, coreset_indices)

    
    coreset_loader = DataLoader(coreset_dataset, batch_size=128, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20(num_classes=10).to(device)

    
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    num_pruning_rounds = 7

    global_start_time = time.time()

    for round in range(num_pruning_rounds):
        print(f"\n--- Pruning Round {round+1}/{num_pruning_rounds} ---")
        round_start_time = time.time()

        for epoch in tqdm(range(num_epochs), desc=f"Prune Round {round+1} Training"):
            train_loss, train_acc = train(
                model, tqdm(coreset_loader, leave=False), optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            tqdm.write(
                f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if "shortcut" in name:  
                    continue
                parameters_to_prune.append((module, "weight"))

        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )

        print_sparsity(model)
        verify_excluded_layers(model)

        round_elapsed = time.time() - round_start_time
        print(
            f"--> Round {round+1} completed in {round_elapsed:.2f} sec ({round_elapsed/60:.2f} min)"
        )

    total_elapsed = time.time() - global_start_time
    print(
        f"\nIterative pruning completed in {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} min)."
    )

    
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal Pruned Model Performance:")
    print(
        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )

    
    
    
    print("\n--- Testing Lottery Ticket Hypothesis ---")
    lottery_model = ResNet20(num_classes=10).to(device)
    
    

    
    for (name, module), (lottery_name, lottery_module) in zip(
        model.named_modules(), lottery_model.named_modules()
    ):
        if hasattr(module, "weight_mask"):
            prune.custom_from_mask(
                lottery_module, name="weight", mask=module.weight_mask
            )

    optimizer = optim.Adam(lottery_model.parameters(), lr=1e-3)

    num_epochs = 50
    for epoch in tqdm(range(num_epochs), desc="Lottery Ticket Training"):
        train_loss, train_acc = train(
            lottery_model, tqdm(train_loader, leave=False), optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(lottery_model, val_loader, criterion, device)
        tqdm.write(
            f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    train_loss, train_acc = evaluate(lottery_model, train_loader, criterion, device)
    val_loss, val_acc = evaluate(lottery_model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(lottery_model, test_loader, criterion, device)

    print("\nLottery Ticket Final Performance:")
    print(
        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Pruning with Lottery Ticket Hypothesis (ResNet20 CIFAR10)")
    parser.add_argument(
        "--percent",
        type=float,
        default=0.2,
        help="Fraction of training data to use (0 < percent <= 1)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["e", "h"],
        default="e",
        help="Select 'e' for easy examples or 'h' for hard examples",
    )
    args = parser.parse_args()

    main(args)
