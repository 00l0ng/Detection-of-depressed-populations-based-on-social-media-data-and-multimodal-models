import jieba
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from transformers import XLNetTokenizer, XLNetModel
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载停用词
stopwords = pd.read_csv("stopwords.txt", encoding="utf-8")
stopwords = stopwords.dropna()
sw = stopwords.stopword.values.tolist()
jieba.load_userdict(r"分词词典.txt")

# 定义对列表进行分词并去除停用词的函数
def seg_rem(data_list):
    clean_final = []
    for line in data_list:
        line = re.sub(r"\s+", "", line)  # 删除文本中的空格、换行符
        line = re.sub(r"\u200b", "", line)
        line = re.sub(r"#\S+#", "", line)
        line = re.sub(r"$$.*? $$", "", line)
        seglist = jieba.lcut(line, cut_all=False)  # 精确模式
        final = []  # 存储去除停用词内容
        for seg in seglist:
            if seg not in sw and len(seg) > 1:
                final.append(seg)
        output = " ".join(final)  # 空格拼接
        clean_final.append(output)
    return clean_final

# 因子化双线性池化层
class FactorizedBilinearPooling(nn.Module):
    def __init__(self, text_feature_dim, img_feature_dim, hidden_dim):
        super(FactorizedBilinearPooling, self).__init__()
        self.text_fc = nn.Linear(text_feature_dim, hidden_dim)
        self.img_fc = nn.Linear(img_feature_dim, hidden_dim)
        self.behavior_fc = nn.Linear(hidden_dim, hidden_dim)  # 新增行为特征的全连接层
        self.bilinear_fc = nn.Linear(hidden_dim, 1)

    def forward(self, text_features, img_features, behavior_features):
        text_out = self.text_fc(text_features)
        img_out = self.img_fc(img_features)
        behavior_out = self.behavior_fc(behavior_features)
        bilinear_out = torch.matmul(text_out.unsqueeze(2), img_out.unsqueeze(1)).sum(dim=2)
        bilinear_out = torch.matmul(behavior_out.unsqueeze(2), bilinear_out.unsqueeze(1)).sum(dim=2)
        output = torch.sigmoid(self.bilinear_fc(bilinear_out))
        return output


# 注意力机制
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 对三个模态的拼接特征计算注意力
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),  # 输出3个权重（对应文本/图像/行为）
            nn.Softmax(dim=1)
        )

    def forward(self, text, img, behavior):
        combined = torch.cat([text, img, behavior], dim=1)  # shape: (batch, hidden*3)
        weights = self.attention(combined)  # shape: (batch, 3)
        # 扩展权重以匹配原始维度
        text_weighted = text * weights[:, 0].unsqueeze(1)
        img_weighted = img * weights[:, 1].unsqueeze(1)
        behavior_weighted = behavior * weights[:, 2].unsqueeze(1)
        return text_weighted, img_weighted, behavior_weighted, weights

# 多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, text_feature_dim, img_feature_dim, behavior_feature_dim, hidden_dim, dropout_rate=0.2):
        super(MultiModalModel, self).__init__()
        # 特征转换层
        self.text_fc = nn.Linear(text_feature_dim, hidden_dim)
        self.img_fc = nn.Linear(img_feature_dim, hidden_dim)
        self.behavior_fc = nn.Linear(behavior_feature_dim, hidden_dim)

        # 注意力机制
        self.cross_attention = CrossModalAttention(hidden_dim)

        # 融合层
        self.fusion_fc = nn.Linear(hidden_dim * 3, 1)
        self.factorized_bilinear = FactorizedBilinearPooling(hidden_dim, hidden_dim, hidden_dim)

        # 正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 添加层归一化

    def forward(self, text_features, img_features, behavior_features):
        # 特征投影
        text_out = self.layer_norm(torch.relu(self.text_fc(text_features)))
        img_out = self.layer_norm(torch.relu(self.img_fc(img_features)))
        behavior_out = self.layer_norm(torch.relu(self.behavior_fc(behavior_features)))

        # 应用Dropout
        text_out = self.dropout(text_out)
        img_out = self.dropout(img_out)
        behavior_out = self.dropout(behavior_out)

        # 跨模态注意力
        text_w, img_w, behavior_w, attn_weights = self.cross_attention(
            text_out, img_out, behavior_out
        )

        # 特征融合
        fused = torch.cat((text_w, img_w, behavior_w), dim=1)
        fused = self.dropout(fused)

        # 双分支输出
        output_main = torch.sigmoid(self.fusion_fc(fused))
        output_bilinear = self.factorized_bilinear(text_w, img_w, behavior_w)

        return {
            "output": output_main,
            "output_bilinear": output_bilinear,
            "attention_weights": attn_weights  # 形状(batch_size, 3)
        }


def select_users_from_images(image_folder, num_users=1289):
    image_files = os.listdir(image_folder)
    user_indices = [int(filename.split("_")[1].split(".")[0]) for filename in image_files]
    unique_users = list(set(user_indices))
    print(len(unique_users))
    selected_users = random.sample(unique_users, num_users)
    return selected_users

def load_data(csv_file_path, image_folder, selected_users, label):
    data = pd.read_csv(csv_file_path, encoding="utf-8")
    data = data.dropna(subset=["tweet_content"])
    data = data[data["tweet_content"] != "无"]
    data["tweet_content"] = data["tweet_content"].astype(str)
    data_labels = []
    data_text_features = []
    data_img_features = []
    data_behavior_features = []

    # XLNet相关设置
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
    xlnet = XLNetModel.from_pretrained("hfl/chinese-xlnet-base")
    xlnet.eval()
    # ResNet50相关设置
    resnet = resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
    resnet.eval()
    # 定义图像增强和预处理的转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义图像特征提取函数
    def extract_image_features(image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = resnet(image_tensor)
        return features.squeeze(0).squeeze(-1).squeeze(-1)

    user_features = {}
    processed_users = set()
    for index, row in data.iterrows():
        user_index = row["user_index"]
        tweet_sub_index = row["tweet_sub_index"]

        if user_index in selected_users and user_index not in processed_users:
            tweet_text = row["tweet_content"]
            clean_text = seg_rem([tweet_text])
            encoded_input = tokenizer(clean_text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                text_feature = xlnet(**encoded_input).last_hidden_state.mean(dim=1)

            image_files = [f for f in os.listdir(image_folder) if f.startswith(f"user_{user_index}_")]
            if image_files:
                image_files = sorted(image_files)
                if user_index not in user_features:
                    user_features[user_index] = {
                        "text_features": [],
                        "img_features": [],
                        "behavior_features": [],
                        "label": label
                    }
                for selected_image_file in image_files[:5]:
                    img_path = os.path.join(image_folder, selected_image_file)
                    if os.path.exists(img_path):
                        img_feature = extract_image_features(img_path)
                        user_features[user_index]["text_features"].append(text_feature)
                        user_features[user_index]["img_features"].append(img_feature)
                        user_features[user_index]["behavior_features"].append(torch.tensor([
                            row["num_of_follower"],
                            row["num_of_following"],
                            row["all_tweet_count"],
                            row["original_tweet_count"],
                            row["repost_tweet_count"],
                            row['num_of_likes'],
                            row['tweet_is_original']
                        ], dtype=torch.float32))
                    else:
                        print(f"Image file not found: {img_path}")
            else:
                print(f"No image files found for user index: {user_index}")

            processed_users.add(user_index)

    for user_index, features in user_features.items():
        text_features_tensor = torch.stack(features["text_features"])
        avg_text_feature = torch.mean(text_features_tensor, dim=0)
        img_features_tensor = torch.stack(features["img_features"])
        avg_img_feature = torch.mean(img_features_tensor, dim=0)
        behavior_features_tensor = torch.stack(features["behavior_features"])
        avg_behavior_feature = torch.mean(behavior_features_tensor, dim=0)

        data_text_features.append(avg_text_feature)
        data_img_features.append(avg_img_feature)
        data_behavior_features.append(avg_behavior_feature)
        data_labels.append(features["label"])

    return data_text_features, data_img_features, data_behavior_features, data_labels


depressed_image_folder = "depressed_images"
normal_image_folder = "normal_images"
depressed_csv_file_path = "merged_depressed.csv"
normal_csv_file_path = "merged_normal.csv"

depressed_selected_users = select_users_from_images(depressed_image_folder)
normal_selected_users = select_users_from_images(normal_image_folder)

depressed_text_features, depressed_img_features, depressed_behavior_features, depressed_labels = load_data(
    depressed_csv_file_path, depressed_image_folder, depressed_selected_users, label=1
)
normal_text_features, normal_img_features, normal_behavior_features, normal_labels = load_data(
    normal_csv_file_path, normal_image_folder, normal_selected_users, label=0
)

# 筛选数据
filtered_text_features = []
filtered_img_features = []
filtered_behavior_features = []
filtered_labels = []
for i in range(len(depressed_labels)):
    filtered_text_features.append(depressed_text_features[i])
    filtered_img_features.append(depressed_img_features[i])
    filtered_behavior_features.append(depressed_behavior_features[i])
    filtered_labels.append(1)
for i in range(len(normal_labels)):
    filtered_text_features.append(normal_text_features[i])
    filtered_img_features.append(normal_img_features[i])
    filtered_behavior_features.append(normal_behavior_features[i])
    filtered_labels.append(0)

# 将筛选后的数据转换为张量
filtered_text_features = torch.stack(filtered_text_features).squeeze(1)  # 展平文本特征的维度
filtered_img_features = torch.stack(filtered_img_features)
filtered_behavior_features = torch.stack(filtered_behavior_features)
filtered_labels = torch.tensor(filtered_labels, dtype=torch.float32).view(-1, 1)

# 对文本特征、图像特征和行为特征进行标准化
def normalize_features(feature_tensor):
    mean = torch.mean(feature_tensor, dim=0)
    std = torch.std(feature_tensor, dim=0)
    normalized_features = (feature_tensor - mean) / (std + 1e-8)
    return normalized_features

filtered_text_features = normalize_features(filtered_text_features)
filtered_img_features = normalize_features(filtered_img_features)
filtered_behavior_features = normalize_features(filtered_behavior_features)

# 使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filtered_text_features = filtered_text_features.to(device)
filtered_img_features = filtered_img_features.to(device)
filtered_behavior_features = filtered_behavior_features.to(device)
filtered_labels = filtered_labels.to(device)

# 划分训练集、验证集和测试集
train_text_features, val_text_features, train_img_features, val_img_features, train_behavior_features, val_behavior_features, train_labels, val_labels = train_test_split(
    filtered_text_features, filtered_img_features, filtered_behavior_features, filtered_labels, test_size=0.2, random_state=42
)
train_text_features, test_text_features, train_img_features, test_img_features, train_behavior_features, test_behavior_features, train_labels, test_labels = train_test_split(
    train_text_features, train_img_features, train_behavior_features, train_labels, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(train_text_features, train_img_features, train_behavior_features, train_labels)
val_dataset = TensorDataset(val_text_features, val_img_features, val_behavior_features, val_labels)
test_dataset = TensorDataset(test_text_features, test_img_features, test_behavior_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 初始化模型、损失函数和优化器
text_feature_dim = 768
img_feature_dim = 2048
behavior_feature_dim = 7
hidden_dim = 256

model = MultiModalModel(text_feature_dim, img_feature_dim, behavior_feature_dim, hidden_dim)
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 早停策略
best_val_loss = float('inf')
patience = 50
early_stop_counter = 0

# 在训练循环中记录注意力权重
attention_weights_list = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for text_batch, img_batch, behavior_batch, label_batch in train_loader:
        text_batch, img_batch, behavior_batch, label_batch = text_batch.to(device), img_batch.to(device), behavior_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        # 修复后的代码
        outputs_dict = model(text_batch, img_batch, behavior_batch)
        outputs = outputs_dict["output"]
        outputs_bilinear = outputs_dict["output_bilinear"]
        attn_weights = outputs_dict["attention_weights"]  # 获取权重

        # 记录注意力权重
        attention_weights_list.append(attn_weights.cpu().detach().numpy())

        loss = criterion(outputs, label_batch) + criterion(outputs_bilinear, label_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        train_total += label_batch.size(0)
        train_correct += (predicted == label_batch).sum().item()

    train_accuracy = train_correct / train_total
    train_loss /= len(train_loader)

    # 验证集评估
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for text_batch, img_batch, behavior_batch, label_batch in val_loader:
            text_batch, img_batch, behavior_batch, label_batch = text_batch.to(device), img_batch.to(device), behavior_batch.to(device), label_batch.to(device)
            outputs_dict = model(text_batch, img_batch, behavior_batch)
            outputs = outputs_dict["output"]
            outputs_bilinear = outputs_dict["output_bilinear"]
            val_loss += criterion(outputs, label_batch).item() + criterion(outputs_bilinear, label_batch).item()
            predicted = (outputs > 0.5).float()
            val_total += label_batch.size(0)
            val_correct += (predicted == label_batch).sum().item()

    val_accuracy = val_correct / val_total
    val_loss /= len(val_loader)

    # 早停策略
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break


# 评价
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for text_batch, img_batch, behavior_batch, label_batch in data_loader:
            text_batch, img_batch, behavior_batch, label_batch = text_batch.to(device), img_batch.to(device), behavior_batch.to(device), label_batch.to(device)
            outputs, _, _, _, _ = model(text_batch, img_batch, behavior_batch)
            predictions = (outputs > 0.5).float()
            all_labels.extend(label_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    return (
        accuracy_score(all_labels, all_predictions),
        precision_score(all_labels, all_predictions),
        recall_score(all_labels, all_predictions),
        f1_score(all_labels, all_predictions)
    )

# 评估验证集
val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1 Score: {val_f1}")

# 评估测试集
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")


