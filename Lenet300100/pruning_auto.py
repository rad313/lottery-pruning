import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import time
from tqdm import tqdm
import argparse





class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)





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
    if total_params > 0:
        total_sparsity = 100.0 * total_zero / total_params
        print(
            f"--> Total Model Sparsity: {total_sparsity:.2f}% pruned ({total_params - total_zero}/{total_params} remain)\n"
        )





def main(args):
    
    transform = transforms.Compose([transforms.ToTensor()])

    base_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_size = int(0.9 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        base_train_dataset, [train_size, val_size]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
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
    model = LeNet().to(device)

    
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    num_pruning_rounds = 10

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

        
        prune.l1_unstructured(model.fc1, name="weight", amount=0.2)
        prune.l1_unstructured(model.fc2, name="weight", amount=0.2)
        prune.l1_unstructured(model.fc3, name="weight", amount=0.1)

        print_sparsity(model)

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
    lottery_model = LeNet().to(device)
    lottery_model.load_state_dict(initial_state_dict)  

    
    for (name, module) in model.named_modules():
        if hasattr(module, "weight_mask"):
            prune.custom_from_mask(
                lottery_model._modules[name], name="weight", mask=module.weight_mask
            )

    optimizer = optim.SGD(lottery_model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 30
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
    parser = argparse.ArgumentParser(description="Iterative Pruning with Lottery Ticket Hypothesis")
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
