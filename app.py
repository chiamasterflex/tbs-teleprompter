import streamlit as st
import time
import queue
import re
import os
import json
import av
import threading

# --- WEB AUDIO ---
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

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 20px; height: 50vh; overflow-y: auto; }
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.6; margin-bottom: 15px; }
    .target-display { font-size: 22px; color: #1a73e8; line-height: 1.6; font-weight: 500; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 10px; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #70757a; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. GLOBAL BUFFERS ---
# Using standard queues to bypass Streamlit's Thread context issues
if 'cn_buf' not in st.session_state: st.session_state.cn_buf = queue.Queue()
if 'en_buf' not in st.session_state: st.session_state.en_buf = queue.Queue()

# --- 4. PERSISTENT STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
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

# --- 5. ENGINE FUNCTIONS ---
def get_aliyun_token():
    client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
    req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
    req.set_version('2019-07-17'); req.set_action_name('CreateToken')
    return json.loads(client.do_action_with_exception(req))['Token']['Id']

def translate_callback_logic(text):
    """Runs in background, pushes result to buffer"""
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if vector_store:
        docs = vector_store.similarity_search(text, k=2)
        prompt += "\nRef: " + " ".join([d.page_content for d in docs])
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.1
        )
        res = response.choices[0].message.content.strip()
        st.session_state.en_buf.put({"cn": text, "en": re.sub(r'[\u4e00-\u9fff]+', '', res)})
    except: pass

# --- 6. THE SAFE CALLBACK ---
def audio_frame_callback(frame: av.AudioFrame):
    # This thread NEVER touches st.session_state directly
    if not hasattr(st.session_state, "_asr_sr"):
        token = get_aliyun_token()
        appkey = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
        
        st.session_state._asr_sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: st.session_state.cn_buf.put(json.loads(m)['payload']['result']),
            on_sentence_end=lambda m, *a: threading.Thread(target=translate_callback_logic, args=(json.loads(m)['payload']['result'],)).start()
        )
        st.session_state._asr_resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state._asr_sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    for r_frame in st.session_state._asr_resampler.resample(frame):
        st.session_state._asr_sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 7. MAIN UI ---
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
    m_in = st.text_area("Type Chinese:", placeholder="Enter text...")
    if st.button("Translate Text"):
        threading.Thread(target=translate_callback_logic, args=(m_in,)).start()

# --- 8. UI REFRESH & BUFFER DRAIN ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='panel-header'>Source</div>", 1)
    src_p = st.empty()
with c2:
    st.markdown("<div class='panel-header'>Translation</div>", 1)
    tar_p = st.empty()

# The Heartbeat loop handles UI updates to avoid "Missing Context"
while webrtc_ctx and webrtc_ctx.state.playing:
    # Drain CN Buffer for live view
    while not st.session_state.cn_buf.empty():
        st.session_state.live_cn = st.session_state.cn_buf.get()
    
    # Drain EN Buffer for history
    while not st.session_state.en_buf.empty():
        st.session_state.history.append(st.session_state.en_buf.get())
        st.session_state.live_cn = ""

    # Render panels
    src_html = "<div class='translation-card'>"
    if st.session_state.live_cn: src_html += f"<div class='text-display live-stream'>{st.session_state.live_cn}</div>"
    for i in reversed(st.session_state.history): src_html += f"<div class='text-display'>{i['cn']}</div><hr>"
    src_p.markdown(src_html + "</div>", 1)

    tar_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.history):
        tar_html += f"<div class='target-display'>{i['en']}</div>"
        tar_html += f"<code style='color:#757575'>{i['en']}</code><hr>"
    tar_p.markdown(tar_html + "</div>", 1)
    
    time.sleep(0.3)

# Static render when not playing
if not webrtc_ctx or not webrtc_ctx.state.playing:
    if hasattr(st.session_state, "_asr_sr"):
        try: st.session_state._asr_sr.stop()
        except: pass
        del st.session_state._asr_sr

    src_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.history): src_html += f"<div class='text-display'>{i['cn']}</div><hr>"
    src_p.markdown(src_html + "</div>", 1)

    tar_html = "<div class='translation-card'>"
    for i in reversed(st.session_state.history):
        tar_html += f"<div class='target-display'>{i['en']}</div>"
        tar_html += f"<code style='color:#757575'>{i['en']}</code><hr>"
    tar_p.markdown(tar_html + "</div>", 1)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain Status: {'Online' if vector_store else 'Offline'}")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
