import streamlit as st
import time
import queue
import re
import os
import json
import threading

# --- WEB AUDIO ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- ALIBABA & OPENAI ---
import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from openai import OpenAI

# --- RAG ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. BOOTSTRAP (Safety First) ---
try:
    ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
    ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    st.error("Secrets not found in Streamlit Dashboard.")
    st.stop()

ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# --- 2. UI LAYOUT (Loads instantly) ---
st.set_page_config(page_title="TBS Pro Translator", layout="wide")
st.title("🪷 TBS Pro Translator")

# Persistent State
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
if 'dialect' not in st.session_state: st.session_state.dialect = "Mandarin"
if 'last_latency' not in st.session_state: st.session_state.last_latency = 0.0

# --- 3. LAZY LOADING BRAIN (Does not run on boot) ---
@st.cache_resource
def get_vector_store():
    # This only runs when actually needed, preventing startup crashes
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if os.path.exists(DB_PATH):
            return FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)
    except: return None
    return None

# --- 4. ENGINE FUNCTIONS ---
def get_aliyun_token():
    client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
    req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
    req.set_version('2019-07-17'); req.set_action_name('CreateToken')
    return json.loads(client.do_action_with_exception(req))['Token']['Id']

def translate_and_commit(text):
    start_t = time.time()
    brain = get_vector_store()
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if brain:
        docs = brain.similarity_search(text, k=2)
        prompt += "\nContext: " + " ".join([d.page_content for d in docs])
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        temperature=0.1
    )
    res_text = response.choices[0].message.content
    st.session_state.last_latency = round(time.time() - start_t, 2)
    st.session_state.history.append({"cn": text, "en": res_text.strip(), "lat": st.session_state.last_latency})

# --- 5. AUDIO CALLBACK (Python 3.13 Safe) ---
def audio_frame_callback(frame: av.AudioFrame):
    if "sr" not in st.session_state:
        appkey = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
        token = get_aliyun_token()
        st.session_state.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(st.session_state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=lambda m, *a: translate_and_commit(json.loads(m)['payload']['result'])
        )
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    for r_frame in st.session_state.resampler.resample(frame):
        st.session_state.sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 6. UI TABS ---
tab_v, tab_m = st.tabs(["🎙️ Voice Mode", "⌨️ Manual Mode"])

with tab_v:
    webrtc_ctx = webrtc_streamer(
        key="asr",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback,
    )

with tab_m:
    m_input = st.text_area("Type Chinese:")
    if st.button("Translate Text"): translate_and_commit(m_input)

# Cleanup
if webrtc_ctx and not webrtc_ctx.state.playing and "sr" in st.session_state:
    try: st.session_state.sr.stop()
    except: pass
    del st.session_state.sr

st.divider()

# --- 7. THE GOOGLE-STYLE DISPLAY ---
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### Source")
    if st.session_state.live_cn: st.info(st.session_state.live_cn)
    for i in reversed(st.session_state.history): 
        st.write(i['cn'])
        st.divider()

with col_right:
    st.markdown("### Translation")
    for i in reversed(st.session_state.history):
        st.success(i['en'])
        st.code(i['en'], language="text") # Google Style Copy Icon
        st.caption(f"Latency: {i['lat']}s")
        st.divider()

# Sidebar Setup
with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    # Brain only initializes if someone views the sidebar
    brain = get_vector_store()
    st.write(f"🧠 Brain Status: {'Loaded' if brain else 'Empty'}")
    if brain: st.write(f"Items: {brain.index.ntotal}")
    if st.button("Clear Session"): st.session_state.history = []; st.rerun()

# Keep live text updating
if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.1)
    st.rerun()
