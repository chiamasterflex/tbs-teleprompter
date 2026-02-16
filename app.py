import streamlit as st
import time
import re
import os
import json
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- ALIBABA & OPENAI ---
import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from openai import OpenAI

# --- RAG ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 2. CSS STYLING (Google Translate Style) ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 24px; height: 50vh; overflow-y: auto; box-shadow: 0 1px 2px rgba(60,64,67,0.3); }
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.6; margin-bottom: 20px; }
    .target-display { font-size: 24px; color: #1a73e8; line-height: 1.6; font-weight: 400; margin-bottom: 20px; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 4px solid #e67e22; padding-left: 12px; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #70757a; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. THREAD-SAFE GLOBAL BUFFERS ---
# These do NOT belong to st.session_state to avoid Context errors
if 'SHARED_HISTORY' not in st.session_state: st.session_state.SHARED_HISTORY = []
if 'LIVE_CN_TEXT' not in st.session_state: st.session_state.LIVE_CN_TEXT = ""

# --- 4. PERSISTENT STATE ---
if 'history' not in st.session_state: st.session_state.history = []
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

# --- 5. TRANSLATION LOGIC ---
def translate_task(text):
    """Isolated translation task"""
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if vector_store:
        try:
            docs = vector_store.similarity_search(text, k=2)
            prompt += "\nRef: " + " ".join([d.page_content for d in docs])
        except: pass
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.1
        )
        res = response.choices[0].message.content.strip()
        clean_res = re.sub(r'[\u4e00-\u9fff]+', '', res)
        # Push to the shared list
        st.session_state.SHARED_HISTORY.append({"cn": text, "en": clean_res})
    except: pass

# --- 6. THE FIREWALLED CALLBACK ---
def audio_frame_callback(frame: av.AudioFrame):
    # This thread is totally isolated. It uses local variables for SR.
    # It does NOT use st.session_state or st.write.
    
    # Check if SR initialized in this thread's scope
    if not hasattr(audio_frame_callback, "sr"):
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        req.set_version('2019-07-17'); req.set_action_name('CreateToken')
        token = json.loads(client.do_action_with_exception(req))['Token']['Id']
        
        appkey = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE

        def on_sentence_end(message, *args):
            res = json.loads(message)['payload']['result']
            # Kick off translation in a new thread so this one doesn't hang
            threading.Thread(target=translate_task, args=(res,), daemon=True).start()

        audio_frame_callback.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(st.session_state, 'LIVE_CN_TEXT', json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end
        )
        audio_frame_callback.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        audio_frame_callback.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    # Resample and send
    for r_frame in audio_frame_callback.resampler.resample(frame):
        audio_frame_callback.sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 7. UI LAYOUT ---
st.title("🪷 TBS Pro Translator")

tab_v, tab_m = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab_v:
    webrtc_ctx = webrtc_streamer(
        key="asr",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback,
    )

with tab_m:
    m_in = st.text_area("Type Chinese:", height=100)
    if st.button("Translate Manual"):
        threading.Thread(target=translate_task, args=(m_in,), daemon=True).start()

st.divider()

# --- 8. UI HEARTBEAT (The safe renderer) ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='panel-header'>Source</div>", 1)
    src_p = st.empty()
with col2:
    st.markdown("<div class='panel-header'>English</div>", 1)
    tar_p = st.empty()

# Heartbeat renders current state every 500ms
while webrtc_ctx and webrtc_ctx.state.playing:
    # 1. Update Chinese
    src_html = "<div class='translation-card'>"
    if st.session_state.LIVE_CN_TEXT:
        src_html += f"<div class='text-display live-stream'>{st.session_state.LIVE_CN_TEXT}</div>"
    for i in reversed(st.session_state.SHARED_HISTORY):
        src_html += f"<div class='text-display'>{i['cn']}</div><hr>"
    src_p.markdown(src_html + "</div>", unsafe_allow_html=True)

    # 2. Update English
    tar_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.SHARED_HISTORY):
        tar_html += f"<div class='target-display'>{i['en']}</div>"
        tar_html += f"<code style='color:#757575'>{i['en']}</code><hr>"
    tar_p.markdown(tar_html + "</div>", unsafe_allow_html=True)
    
    time.sleep(0.5)

# Final render if stopped
if not webrtc_ctx or not webrtc_ctx.state.playing:
    # Cleanup SR if it was left behind in the callback attribute
    if hasattr(audio_frame_callback, "sr"):
        try: audio_frame_callback.sr.stop()
        except: pass
        del audio_frame_callback.sr

    # Static display of history
    src_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.SHARED_HISTORY):
        src_html += f"<div class='text-display'>{i['cn']}</div><hr>"
    src_p.markdown(src_html + "</div>", 1)

    tar_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.SHARED_HISTORY):
        tar_html += f"<div class='target-display'>{i['en']}</div>"
        tar_html += f"<code style='color:#757575'>{i['en']}</code><hr>"
    tar_p.markdown(tar_html + "</div>", 1)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain Status: {'Online' if vector_store else 'Offline'}")
    if st.button("Clear History"):
        st.session_state.SHARED_HISTORY = []
        st.session_state.LIVE_CN_TEXT = ""
        st.rerun()
