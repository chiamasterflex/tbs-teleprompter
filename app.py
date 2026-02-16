import streamlit as st
import time
import threading
import queue
import re
import os
import json
import av

# --- WEB AUDIO ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit.runtime.scriptrunner import add_script_run_context

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

# --- 1. SECRETS & CONFIG ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 2. CSS ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 20px; height: 50vh; overflow-y: auto; }
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.6; margin-bottom: 15px; }
    .target-display { font-size: 22px; color: #1a73e8; line-height: 1.6; font-weight: 500; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
if 'live_en' not in st.session_state: st.session_state.live_en = ""
if 'run' not in st.session_state: st.session_state.run = False
if 'dialect' not in st.session_state: st.session_state.dialect = "Mandarin"

@st.cache_resource
def get_brain():
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if os.path.exists(DB_PATH):
            return FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)
    except: return None
    return None

vector_store = get_brain()

# --- 4. TRANSLATION ENGINE ---
def run_translation(text, is_final):
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if vector_store:
        docs = vector_store.similarity_search(text, k=2)
        prompt += "\nRef: " + " ".join([d.page_content for d in docs])
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            stream=True, temperature=0.1
        )
        full_res = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content: 
                full_res += content
                if not is_final: st.session_state.live_en = full_res + "..."
        
        if is_final:
            st.session_state.history.append({"cn": text, "en": full_res.strip()})
            st.session_state.live_en = ""
    except: pass

# --- 5. ALIYUN THREADED ENGINE ---
def start_aliyun(webrtc_ctx, appkey):
    # This function is now Script-Context aware
    try:
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        req.set_version('2019-07-17'); req.set_action_name('CreateToken')
        token = json.loads(client.do_action_with_exception(req))['Token']['Id']
        
        def on_sentence_end(message, *args):
            txt = json.loads(message)['payload']['result']
            run_translation(txt, True)
            st.session_state.live_cn = ""

        sr = nls.NlsSpeechTranscriber(url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(st.session_state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end)
        
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})
        
        while st.session_state.run and webrtc_ctx.state.playing:
            if webrtc_ctx.audio_receiver:
                try:
                    frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                    for frame in frames:
                        for r_frame in resampler.resample(frame):
                            sr.send_audio(r_frame.to_ndarray().tobytes())
                except: break
            else: time.sleep(0.1)
        sr.stop()
    except: pass

# --- 6. UI TABS ---
st.title("🪷 TBS Pro Translator")

tab1, tab2 = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab1:
    active_key = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
    webrtc_ctx = webrtc_streamer(
        key="asr", mode=WebRtcMode.SENDONLY, 
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True, audio_receiver_size=1024
    )

with tab2:
    m_in = st.text_area("Type Chinese:", height=100)
    if st.button("Translate Now"): run_translation(m_in, True)

# THREAD MANAGEMENT
if webrtc_ctx.state.playing and not st.session_state.run:
    st.session_state.run = True
    t = threading.Thread(target=start_aliyun, args=(webrtc_ctx, active_key))
    # CRITICAL: Attach the thread to the Streamlit UI context
    add_script_run_context(t) 
    t.start()
elif not webrtc_ctx.state.playing and st.session_state.run:
    st.session_state.run = False

st.divider()

# --- 7. DISPLAY PANELS ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("Source")
    src_h = "<div class='translation-card'>"
    if st.session_state.live_cn: src_h += f"<div class='text-display live-stream'>{st.session_state.live_cn}</div>"
    for i in reversed(st.session_state.history): src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", 1)
with c2:
    st.subheader("English")
    tar_h = "<div class='translation-card'>"
    if st.session_state.live_en: tar_h += f"<div class='target-display live-stream'>{st.session_state.live_en}</div>"
    for i in reversed(st.session_state.history):
        st.markdown(f"<div class='target-display'>{i['en']}</div>", 1)
        st.code(i['en'], language="text")
        st.divider()
    st.markdown("</div>", 1)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain: {vector_store.index.ntotal if vector_store else 0} items")
    if st.button("Clear History"): st.session_state.history = []; st.rerun()

if st.session_state.run:
    time.sleep(0.4)
    st.rerun()
