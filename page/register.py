import sys
import time
sys.path.append("..")

import cv2
import numpy as np
import streamlit as st
import hydralit as hy
from PIL import Image 
from datetime import datetime
import torch
from torchvision import transforms

from utils.functions import extract_features, density_filter, WarnInFrame
from model.GEINET import GEINet2

def get_state():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'vectors_rec' not in st.session_state:
        st.session_state.vectors_rec = []
    if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
    if "name_rec" not in st.session_state:
        st.session_state.name = ""
    if "register" not in st.session_state:
        st.session_state.register = False
    if "avgvec_rec" not in st.session_state:
        st.session_state.avgvec_rec = None
    if "gei_rec" not in st.session_state:
        st.session_state.gei_rec = None
    if "date_rec" not in st.session_state:
        st.session_state.date_rec = None
    if "submit" not in st.session_state:
        st.session_state.submit = False


def record(camera_st):
    start_time = time.time()

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
        # Read a frame from the video
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
        
        #ベクトル抽出
        gei_tensor = transforms.ToTensor()(gei).to(device)
        features = extract_features(model_gait, gei_tensor.reshape(1,1,128,88), -3)
        vec = features.numpy()
        vec = vec.squeeze()
        vec = vec / np.linalg.norm(vec)       # 大きさを1にそろえる

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

        # REC
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (117, 91, 255), 20)
        # 5秒間の平均ベクトルを計算
        if time.time() - start_time < 5:
            st.session_state.vectors_rec.append(vec)  # ベクトルを追加
        else:
            if st.session_state.vectors_rec:
                # 平均ベクトルを計算
                avg_vector = np.mean(st.session_state.vectors_rec, axis=0)                
                return avg_vector, gei

        frame_st = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera_st.image(frame_st)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

def normal_camera(camera_st):
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
        # Read a frame from the video
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
        
        #ベクトル抽出
        gei_tensor = transforms.ToTensor()(gei).to(device)
        features = extract_features(model_gait, gei_tensor.reshape(1,1,128,88), -3)
        vec = features.numpy()
        vec = vec.squeeze()
        vec = vec / np.linalg.norm(vec)       # 大きさを1にそろえる

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

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
                
def register():
    get_state()

    col1, col2 = st.columns([1, 1])
    with col1:
        camera_st = st.empty()
    with col2:
        # button_st = st.empty()
        info_st = st.empty()
        form_st = st.empty()
        submit_form_st = st.empty()
        
    if not st.session_state.recording:
        with form_st.form("my_form", clear_on_submit=True):
            st.session_state.name_rec = st.text_input("name")
            if st.form_submit_button("REC"):
                st.session_state.recording = True
                st.rerun()
                
    if st.session_state.recording:
        info_st.info("recording...")
        avgvec_rec, gei_rec = record(camera_st=camera_st)
        st.session_state.avgvec_rec = avgvec_rec / np.linalg.norm(avgvec_rec)
        st.session_state.gei_rec = cv2.resize(gei_rec, (int(gei_rec.shape[1]*2.5), int(gei_rec.shape[0]*2.5)))
        st.session_state.vectors_rec = []
        st.session_state.recording = False
        st.session_state.register = True
    
    if st.session_state.register:
        with submit_form_st.form("submit_form", clear_on_submit=True):
            with st.container():
                col1_sub, col2_sub = st.columns([1, 2])
                with col1_sub:
                    st.text(f"name: {st.session_state.name_rec}")
                    st.session_state.date_rec = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
                    st.text(f"date: {st.session_state.date_rec}")
                with col2_sub:
                    st.image(st.session_state.gei_rec, caption="GEI")
            if st.form_submit_button("SUBMIT"):
                st.session_state.submit = True
                st.session_state.register = False
                st.rerun()
    else:
        submit_form_st.empty()

    if st.session_state.submit:
        # データベースへの保存作業
        np.savetxt(f"./data/vectors/{st.session_state.name_rec}.txt", st.session_state.avgvec_rec, delimiter=',')
        cv2.imwrite(f"./data/images/{st.session_state.name_rec}.png", st.session_state.gei_rec)
        with open("./data/member_info.txt", "a") as f:
            f.write(f"{st.session_state.name_rec}, {st.session_state.date_rec}\n")
        st.info("register completed!")
        st.session_state.submit = False
        st.session_state.date_rec = None
        st.session_state.name_rec = None
        st.session_state.avgvec_rec = None
        st.session_state.gei_rec = None
    
    if st.session_state.register and not st.session_state.submit:
        info_st.info("press SUBMIT button")
    else:
        info_st.info("enter your name and press REC button")
    normal_camera(camera_st=camera_st)


if __name__ == "__main__":
    register()