import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt





class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        self.model = models.resnet18(weights=None)  
        
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)





class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)





transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

base_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_dataset = IndexedDataset(base_train_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

n_train = len(train_dataset)
prev_acc = torch.zeros(n_train, dtype=torch.long)          
forgetting_count = torch.zeros(n_train, dtype=torch.long)  





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18CIFAR().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 30  

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        
        outputs = model(data)
        preds = outputs.argmax(dim=1)

        
        acc = (preds == target).long().cpu()

        for i, idx in enumerate(indices):
            if prev_acc[idx] == 1 and acc[i] == 0:
                forgetting_count[idx] += 1
            prev_acc[idx] = acc[i]

        
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} complete")



save_path = "forgetting_scores.pt"
torch.save(forgetting_count, save_path)
print(f"Forgetting scores saved to {save_path}")





print("Top 10 most forgotten examples (index, count):")
forgotten_examples = torch.topk(forgetting_count, 10)
for idx, count in zip(forgotten_examples.indices, forgotten_examples.values):
    print(f"Example {idx.item()} forgotten {count.item()} times")




plt.figure(figsize=(8, 5))
plt.hist(
    forgetting_count.numpy(),
    bins=range(forgetting_count.max().item() + 2),
    edgecolor='black',
    align='left'
)
plt.xlabel("Number of forgetting events")
plt.ylabel("Number of training examples")
plt.title("Distribution of Forgetting Events (CIFAR10, ResNet18)")


plot_path = "forgetting_histogram.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Histogram saved to {plot_path}")

plt.show()
