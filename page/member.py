import sys
sys.path.append("..")

import streamlit as st
import pandas as pd

def member():
    members = []
    dates = []

    with open("./data/member_info.txt", "r") as f:
        for line in f:
            name, date = line.strip().split(', ')
            members.append(name)
            dates.append(date)

    # メンバーの情報をPandasのDataFrameに変換
    member_info_df = pd.DataFrame({
        'Name': members,
        'Registraion Date': dates
    })

    # Streamlitで表を表示
    st.table(member_info_df)

