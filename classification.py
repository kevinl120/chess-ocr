import torch
import cv2
import numpy as np

def load_data():
    x = []
    y = []
    base_path = './data/classification'
    images_path = base_path + '/images/'
    labels_path = base_path + '/labels.csv'
    for line in open(labels_path):
        line = line.strip().split(',')
        img = cv2.imread(images_path + line[0], cv2.IMREAD_GRAYSCALE)
        x.append(img)
        y.append(int(line[1]))
    return np.array(x), np.array(y)

def main():
    x, y = load_data()
    dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    train_size = int(0.9 * len(y))
    val_size = len(y) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False) 


    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Linear(64 * 7 * 7, 1024),
        torch.nn.LogSoftmax(dim=1)
    )

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        for x, y in train_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (y_pred.argmax(dim=1) == y).sum().item()
        for x, y in val_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
            val_acc += (y_pred.argmax(dim=1) == y).sum().item()
        print(f"Epoch {epoch+1}: train loss: {train_loss/train_size:.4f}, val loss: {val_loss/val_size:.4f}, train acc: {train_acc/train_size:.4f}, val acc: {val_acc/val_size:.4f}")


main()