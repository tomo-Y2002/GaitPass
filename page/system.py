import sys
import time
sys.path.append("..")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image 
import torch
from torchvision import transforms

from utils.functions import extract_features, density_filter, WarnInFrame
from model.GEINET import GEINet2

def get_state():
    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)

def system():    
    get_state()
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        camera_st = st.empty()
    with col2:
        mask_st = st.empty()
    with col3:
        gei_st = st.empty()

    # グラフ用
    member = []
    DB = []

    with open("./data/member_info.txt", "r") as f:
        for line in f:
            name, _ = line.split(', ')
            member.append(name)
            DB.append(np.loadtxt(f"./data/vectors/{name}.txt", delimiter=","))

    # dot_product_list_size = 1000
    # dot_product_list = np.zeros((len(member), dot_product_list_size), dtype=np.float32)
    dot_product_list = [[] for _ in range(len(member))]
    # dot_product_list_index = 0

    # モデルのパラメタ読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_gait = GEINet2
    model_gait.to(device)
    model_gait.load_state_dict(torch.load("./model/weight/model_0.848.pth", map_location=device))

    # Open the video file
    # video_path = "C:/Users/denjo/Desktop/HackU/data/walk1.mp4"
    # cap = cv2.VideoCapture(0)

    # MOG2背景差分器の作成
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=True)

    # 移動平均のbuffer初期化
    buffer_size = 30
    frame_width = 88
    frame_height = 128
    frame_buffer = np.zeros((buffer_size, frame_height, frame_width), dtype=np.float32)
    buffer_index = 0

    while st.session_state.cap.isOpened():
        
        success, frame = st.session_state.cap.read()
        if not success:
            break

        # 背景差分を適用
        fgMask = backSub.apply(frame)

        # モルフォロジー変換: ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        #密度うめ
        fgMask = density_filter(fgMask, 15, 0.4)
        # fgMask = density_filter(fgMask, 5, 0.4)

        # 輪郭を検出
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最大の輪郭を見つける
        box_x = box_y = box_w = box_h = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500: # 最大の輪郭の面積が500以上の場合
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_x = x
                box_y = y
                box_w = w
                box_h = h
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

                #マスクのクロップ
                cropped_mask = fgMask[y:y+h, x:x+w]
                resized_mask = cv2.resize(cropped_mask, (int(frame_height * float(w) / h), frame_height))
                resized_width = resized_mask.shape[1]

                # 重心の計算
                M = cv2.moments(resized_mask)
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                else:
                    centroid_x = resized_width // 2

                mask_temp = np.zeros((frame_height, frame_width), dtype=np.float32)
                # 重心を中心に配置するための開始X座標を計算
                start_x = frame_width // 2 - centroid_x
                end_x = start_x + resized_mask.shape[1]

                # `resized_mask`を`mask_temp`に配置
                if start_x < 0:  # `resized_mask`が左にはみ出る場合
                    if end_x > frame_width:  # `resized_mask`が右にもはみ出る場合
                        mask_temp[:, :] = resized_mask[:, -start_x:-(start_x - frame_width)]
                    else:  
                        mask_temp[:, :end_x] = resized_mask[:, -start_x:]
                elif end_x > frame_width:  # `resized_mask`が右にはみ出る場合
                    mask_temp[:, start_x:] = resized_mask[:, :(frame_width - start_x)]
                else:  # `resized_mask`が完全に収まる場合
                    mask_temp[:, start_x:end_x] = resized_mask
                
                resized_mask = mask_temp

                frame_buffer[buffer_index % buffer_size] = resized_mask
                buffer_index += 1

        # 結果を表示
        
        gei = np.zeros((frame_height, frame_width), dtype=np.uint8)
        if buffer_index >= buffer_size:
            moving_avg_mask = np.mean(frame_buffer, axis=0).astype(np.uint8)
            gei = moving_avg_mask
            # cv2.imshow('Moving Average Mask', cv2.resize(moving_avg_mask, (frame_width*5, frame_height*5)))
            gei_st.image(cv2.resize(moving_avg_mask, (frame_width*5, frame_height*5)))
        
        #ベクトル抽出
        gei_tensor = transforms.ToTensor()(gei).to(device)
        features = extract_features(model_gait, gei_tensor.reshape(1,1,128,88), -3)
        vec = features.cpu().numpy()
        vec = vec.squeeze()
        vec = vec / np.linalg.norm(vec)       # 大きさを1にそろえる

        # データベースから見つける
        # データベースから見つける        
        normed_DB = np.array(DB) / np.linalg.norm(DB, axis=1, keepdims=True)
        dot_products = np.dot(normed_DB, vec)
        norm_id = np.argmax(dot_products)
        for idx, dot_product in enumerate(dot_products):
            # dot_product_list[idx][dot_product_list_index % dot_product_list_size] = dot_product
            dot_product_list[idx].append(dot_product)
        # dot_product_list_index += 1
        
        name = member[norm_id]
    
        # 名前を枠上に記述
        cv2.putText(frame,
                    text=name,
                    org=(box_x + box_w -100, box_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(123, 0,122),
                    thickness=2,
                    lineType=cv2.LINE_4)       

       
        frame_st = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera_st.image(frame_st)
        mask_st.image(fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

