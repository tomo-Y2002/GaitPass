import streamlit as st
import hydralit as hy

from page.home import home
from page.system import system
from page.register import register
from page.member import member

# streamlitの設定
st.set_page_config(
    page_title="Gait Pass",
    page_icon=":shoe:",
    layout="wide",
    initial_sidebar_state="auto",
)


app = hy.HydraApp(title="Gait Pass")

@app.addapp(is_home=True, title='Home')
def my_home():
    st.title("Gait Pass")
    home()

@app.addapp(title="System")
def app2():
    st.title("System Info")
    system()

@app.addapp(title="Register")
def app3():
    st.title("Register")
    register()

@app.addapp(title="Member")
def app4():
    st.title("Member")
    member()
   
   
app.run()
