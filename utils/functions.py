import cv2
import torch
import numpy as np

def WarnInFrame(frame:np.array, Text:str, width:int, height:int)->None:
    cv2.putText(frame,
                text=Text,
                org=(width//2-220, height//2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(0,255,0),
                thickness=4,
                lineType=cv2.LINE_4)
    
# ベクトル抽出の関数定義
def extract_features(model, input_data, layer_index):
    """
    指定されたモデルの特定の層から特徴量を抽出します。

    :param model: モデル (nn.Module)
    :param input_data: モデルの入力データ (torch.Tensor)
    :param layer_index: 特徴量を取り出したい層のインデックス (int)
    :return: 抽出された特徴量 (torch.Tensor)
    """

    # 出力を保存するリスト
    features = []
    # フック関数
    def hook_fn(module, input, output):
        features.append(output)
    # 指定された層にフックをアタッチ
    hook = model[layer_index].register_forward_hook(hook_fn)
    # モデルを評価モードにする
    model.eval()
    # データをモデルに供給
    with torch.no_grad():
        model(input_data)
    # フックを削除
    hook.remove()
    # 最初の特徴量の出力を返す
    return features[0]

def density_filter(image, kernel_size, threshold):
    # カーネルの作成
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # ローカル密度計算
    local_density = cv2.filter2D(image, -1, kernel) / (kernel_size ** 2)

    # 閾値に基づいてマスクを作成
    mask = local_density > threshold

    # マスクを適用
    return image * mask