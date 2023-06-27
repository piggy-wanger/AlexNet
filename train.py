import os
import sys
import json
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.AlexNet import AlexNet


def main():
    ###############################################################################################
    # 载入数据集
    ###############################################################################################
    # 图片预处理
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    }
    image_path = './data/flower_data'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 载入训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    # 载入验证数据集
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    ###############################################################################################
    # 载入模型
    ###############################################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = AlexNet(num_classes=5)
    net.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    ###############################################################################################
    # 开始训练
    ###############################################################################################
    epochs = 10
    save_path = './weights/AlexNet.pth'
    best_acc = 0.0
    train_step = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_func(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" % (epoch + 1, running_loss / train_step, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished Training!")


if __name__ == '__main__':
    main()

