import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import torch.optim as optim
import os
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import umap
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 5, padding=2, bias = False),
                nn.BatchNorm2d(64),
                nn.GELU()
            )
            self.downsample1 = nn.Sequential(
                nn.Conv2d(3, 64, 1, bias = False),
                nn.BatchNorm2d(64)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride= 2, padding=1, groups=64, bias = False),
                nn.BatchNorm2d(64),
                nn.GELU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 256, 5, padding=2, bias = False),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
            self.downsample2 = nn.Sequential(
                nn.Conv2d(64, 256, 1, bias = False),
                nn.BatchNorm2d(256)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 256,3, stride=2, padding=1, groups=256, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, 256))
            self.pos = nn.Parameter(torch.zeros(1, 8 * 8 + 1, 256))
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=16, activation="gelu",
                                                        batch_first=True, dropout=0.2)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=7)
            self.pre_ln = nn.LayerNorm(256)
            self.head_ln = nn.LayerNorm(256)
            self.fc1 = nn.Linear(256, 512)
            self.drop1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(512, 100)

            nn.init.trunc_normal_(self.cls, std=0.02)
            nn.init.trunc_normal_(self.pos, std=0.02)

        def forward(self, x):
            res1 = self.conv1(x)
            x = self.conv2(self.downsample1(x) + res1)
            res2 = self.conv3(x)
            x = self.conv4(self.downsample2(x) + res2)
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos[:, :x.size(1), :]
            x = self.pre_ln(x)
            x = self.encoder(x)
            cls = x[:, 0]
            mean_tok = x[:, 1:].mean(dim=1)
            x = 0.5 * (cls + mean_tok)
            x = self.head_ln(x)
            x = F.gelu(self.fc1(x))
            x = self.drop1(x)
            x = self.fc2(x)
            return x

        def forward_features(self, x):
            res1 = self.conv1(x)
            x = self.conv2(self.downsample1(x) + res1)
            res2 = self.conv3(x)
            x = self.conv4(self.downsample2(x) + res2)
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos[:, :x.size(1), :]
            x = self.pre_ln(x)
            x = self.encoder(x)
            cls = x[:, 0]
            mean_tok = x[:, 1:].mean(dim=1)
            x = 0.5 * (cls + mean_tok)
            x = self.head_ln(x)
            return x

    def train(epochs_num=5, loss_record = True):
        if loss_record:
            loss_record = []
        mixup_fn = Mixup(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=0.1,
            num_classes=100
        )
        net.train()
        for epoch in range(epochs_num):
            running = 0.0
            epoch_total_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs, labels = mixup_fn(inputs, labels)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running += loss.item()
                epoch_total_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f'[{epoch+1:03d}, {i+1:05d}] loss: {running/100:.3f}')
                    running = 0.0
            loss_record.append(epoch_total_loss / len(trainloader))
        print('Finished Training')
        return loss_record





    def model_SL(PATH, load = False):
        if load == True:
            if os.path.exists(PATH):
                sd = torch.load(PATH, map_location=device)
                net.load_state_dict(sd)  # strict=True
            else:
                print(f'Checkpoint not found: {PATH}')
        else:
            torch.save(net.state_dict(), PATH)

    def loss_curves(loss_record):
        print(loss_record)
        plt.figure()
        plt.plot(loss_record, linestyle='-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.show()


    #show random sample
    def random_sample_show():
        indices = random.sample(range(len(trainset)), 6)

        plt.figure(figsize=(10, 6))
        for i, idx in enumerate(indices):
            image, label = trainset[idx]
            img = image.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = net(img)
                _, predicted = torch.max(outputs, 1)

            vis = (image * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            plt.subplot(2, 3, i + 1)
            plt.imshow(vis)
            plt.title(f'True: {label} | Pred: {predicted.item()}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()

    def test_acc():
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                pred = net(images).argmax(1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return f'Accuracy on test set ({total} images): {100.0 * correct / total:.2f}%'

    def eval_preds():
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device, non_blocking=True)
                logits = net(images)
                pred = logits.argmax(1).cpu()
                ys.append(labels.cpu())
                ps.append(pred)
        y_true = torch.cat(ys)
        y_pred = torch.cat(ps)
        return y_true, y_pred


    def confusion_matrix(y_true, y_pred, num_classes=100):
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm


    def plot_confusion_matrix(cm, class_names=None, normalize=False, fname='confusion_matrix.png', log = False):
        cm_np = cm.numpy().astype(float)
        if normalize:
            row_sums = cm_np.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_np = cm_np / row_sums

        plt.figure(figsize=(10, 8))
        if log == True:
            plt.imshow(np.log(cm_np + 1e9), interpolation='nearest', aspect='auto')
        else:
            plt.imshow(cm_np, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if class_names is not None and len(class_names) == cm_np.shape[0]:
            step = max(1, len(class_names) // 10)
            ticks = np.arange(0, len(class_names), step)
            plt.xticks(ticks, [class_names[i] for i in ticks], rotation=45, ha='right')
            plt.yticks(ticks, [class_names[i] for i in ticks])
        else:
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    def class_accuracies(cm, class_names=None):
        correct = cm.diag().to(torch.float32)
        totals = cm.sum(dim=1).to(torch.float32).clamp(min=1)
        acc = (correct / totals).numpy()
        if class_names is None:
            class_names = [str(i) for i in range(len(acc))]
        rows = [(i, class_names[i], float(acc[i])) for i in range(len(acc))]
        # Print top/bottom quickly
        overall = float(correct.sum() / totals.sum())
        print(f'Overall accuracy from CM: {overall * 100:.2f}%')
        best = sorted(rows, key=lambda x: x[2], reverse=True)[:10]
        worst = sorted(rows, key=lambda x: x[2])[:10]
        print('Top 10 classes:')
        for i, name, a in best:
            print(f'{i:02d} {name:>15}: {a * 100:5.1f}%')
        print('Bottom 10 classes:')
        for i, name, a in worst:
            print(f'{i:02d} {name:>15}: {a * 100:5.1f}%')
        return rows

    def collect_features(dataloader):
        net.eval()
        feats, labels = [], []
        with torch.no_grad():
            for images, y in dataloader:
                images = images.to(device, non_blocking=True)
                z = net.forward_features(images).cpu()
                feats.append(z)
                labels.append(y)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels


    def run_umap_visualization():
        print("UMAP start")
        feats, labels = collect_features(testloader)
        X = StandardScaler().fit_transform(feats.numpy())
        y = labels.numpy()

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", n_jobs=-1)
        emb2d = reducer.fit_transform(X)

        plt.figure()
        plt.scatter(emb2d[:, 0], emb2d[:, 1], c=y, s=4, cmap="tab20")
        plt.tight_layout()
        plt.savefig("umap_test_features.png", dpi=200)
        plt.close()
        print("Saved umap_test_features.png")
        return emb2d, y

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.backends.cudnn.benchmark = True

    batch_size = 128

    CIFAR100_Mean = (0.5071, 0.4866, 0.4409)
    CIFAR100_Std = (0.2673, 0.2564, 0.2762)
    mean_t = torch.tensor(CIFAR100_Mean).view(3, 1, 1)
    std_t = torch.tensor(CIFAR100_Std).view(3, 1, 1)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_Mean, CIFAR100_Std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_Mean, CIFAR100_Std)
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6,
                                              pin_memory=True, persistent_workers=True, prefetch_factor=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6,
                                             pin_memory=True, persistent_workers=True, prefetch_factor=4)

    torch.set_float32_matmul_precision("high")

    net = Net().to(device)
    train_enable = True
    epochs = 200
    criterion = SoftTargetCrossEntropy()
    optimizer = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps_per_epoch = len(trainloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    if train_enable == True:
        loss_list = train(epochs_num=epochs, loss_record=True)
        loss_curves(np.array(loss_list))

    Save_PATH = f'./CIFAR100v2({epochs})_result.pth'
    model_SL(Save_PATH, load = not train_enable)


    net.to(device)
    net.eval()
    random_sample_show()

    y_true, y_pred = eval_preds()
    cm = confusion_matrix(y_true, y_pred, num_classes=100)
    class_names = getattr(testset, 'classes', None)
    plot_confusion_matrix(cm, class_names=class_names, normalize=False, fname='confusion_matrix.png')
    plot_confusion_matrix(cm, class_names=class_names, normalize=True, fname='confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names=class_names, normalize=True, fname='confusion_matrix_normalized.png', log = True)
    per_class = class_accuracies(cm, class_names=class_names)
    run_umap_visualization()