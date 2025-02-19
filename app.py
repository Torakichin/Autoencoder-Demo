import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import streamlit as st
import random

# オートエンコーダモデルのクラス定義
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # デコーダ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 現在のスクリプトがあるディレクトリを取得
current_directory = os.getcwd()

# データセットの準備(CIFAR-10データの取得)
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])

# テストデータの取得
test_dataset = torchvision.datasets.CIFAR10(root=current_directory,  # 現在のディレクトリを指定
                                            train=False,  # テスト用
                                            download=True,  # 必要ならダウンロードする
                                            transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# モデルの読み込み
model = torch.load('autoencoder_model.pth')  # オートエンコーダモデルのファイル名を指定

# モデルを評価モードに設定
model.eval()

# Streamlitアプリ
st.title("Autoencoder Image Reconstruction")

if st.button("Generate Images"):
    # ランダムなテストデータを選択
    random_batch_index = random.randint(0, len(test_dataloader) - 1)
    for i, (inputs, _) in enumerate(test_dataloader):
        if i == random_batch_index:
            break

    # 入力画像と再構成画像を表示
    with torch.no_grad():  # 勾配の計算を無効化
        outputs = model(inputs)
        fig, axes = plt.subplots(10, 2, figsize=(10, 20))
        for i in range(10):
            # オリジナル画像
            axes[i, 0].imshow((inputs[i].permute(1, 2, 0) * 0.5 + 0.5).numpy())
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            # 再構成画像
            axes[i, 1].imshow((outputs[i].permute(1, 2, 0) * 0.5 + 0.5).numpy())
            axes[i, 1].set_title('Reconstructed Image')
            axes[i, 1].axis('off')
        st.pyplot(fig)