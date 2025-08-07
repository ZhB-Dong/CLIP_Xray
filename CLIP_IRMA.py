from Irma import ClassificationDataset, Irma
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import clip

weight_patch = '/home/sda2/dzb/Weights/CLIP_0807/2025-08-07_115948_clip_weight/CLIP_FT1_epoch_49.pth'

categories = {
    "1": "whole body",
    "2": "cranium",
    "3": "spine",
    "4": "upper extremity/arm",
    "5": "chest",
    "6": "breast",
    "7": "abdomen",
    "8": "pelvis",
    "9": "lower extremity",
    # "10": "chest",
    "10": "chest/bones",
    "11": "chest/lung",
    "12": "chest/hilum",
    "13": "chest/mediastinum",
    "14": "chest/heart",
    "15": "chest/diaphragm"
}


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) # 0.02
# model.load_state_dict((torch.load(weight_patch, map_location=device))) # 0.90
irma = Irma('/home/sda2/dzb/Datasets/IRMA/')
irma.load()
data = irma.df
dataset = ClassificationDataset(data, transform=preprocess)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

text_inputs = torch.cat([clip.tokenize(f"a X-Ray image of {c} part") for c in categories.values()]).to(device) # 0.9 0.02

# text_inputs = torch.cat([clip.tokenize(f"a X-Ray image of {c} part in human body") for c in Labels]).to(device) # 0.6049
# text_inputs = torch.cat([clip.tokenize(f"a X-ray of a {c}") for c in Labels]).to(device) # 0.6030
# text_inputs = torch.cat([clip.tokenize(f"This is a X-ray image of {c} part. ") for c in Labels]).to(device) # 0.5887
# text_inputs = torch.cat([clip.tokenize(f"a X-ray image of {c} of human") for c in Labels]).to(device) # 0.5974
# text_inputs = torch.cat([clip.tokenize(f"a image of {c}") for c in Labels]).to(device) # 0.5650

right_count = 0
cnt = 1
# 测试一批数据
for images, labels in dataloader:
    # print(cnt)

    # if cnt > 300:
    #     break
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    # print(images.shape)  # torch.Size([8, 3, 224, 224])
    # print(Labels[labels.item()])        # [['No Finding'], ['Cardiomegaly', 'Effusion'], ...]
    # print('Ground Truth: ' + Labels[labels.item()])
    # print(gene_prompt(labels))
    image_input = images.to(device)
    # print(images.type())
    # image_input = preprocess(images).unsqueeze(0).to(device)
    # print(image_input.size())
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    for indice in indices:
        if labels[0] == categories[str(indice.item()+1)]:
            right_count += 1
    # print(text_features[indices.item()])
    # print(indices)
    # print("Top predictions:")
    # for value, index in zip(values, indices):
    #     print(f"{Labels[index]:>16s}: {100 * value.item():.2f}%")
        # if value.item() in labels:
            # right_count += 1
    # break
print('Right numbers: ', right_count)
print('All numbers: ', cnt)
print('Accuracy: ', right_count/cnt)
