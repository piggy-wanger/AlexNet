import json
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from models.AlexNet import AlexNet


def main():
    ###############################################################################################
    # 图片预处理
    ###############################################################################################
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    image_path = './example/sunflower.jpeg'
    img = Image.open(image_path)

    plt.imshow(img)
    img = data_transform(img)
    image = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    ###############################################################################################
    # 模型加载
    ###############################################################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=5).to(device)

    weight_path = './weights/AlexNet.pth'
    model.load_state_dict(torch.load(weight_path))

    ###############################################################################################
    # 预测
    ###############################################################################################
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
