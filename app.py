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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. GLOBAL BUFFERS (Outside Streamlit State) ---
# These are thread-safe and don't require ScriptRunContext
IF_CN_QUEUE = queue.Queue()
IF_EN_QUEUE = queue.Queue()

# --- 2. CONFIGURATION ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 24px; height: 50vh; overflow-y: auto; }
    .text-display { font-size: 24px; color: #3c4043; line-height: 1.6; margin-bottom: 20px; }
    .target-display { font-size: 24px; color: #1a73e8; line-height: 1.6; font-weight: 500; margin-bottom: 20px; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 4px solid #e67e22; padding-left: 12px; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #70757a; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

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

def translate_task(text):
    """Runs in a thread, pushes to Global EN Queue"""
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
        res_text = response.choices[0].message.content.strip()
        clean_res = re.sub(r'[\u4e00-\u9fff]+', '', res_text)
        IF_EN_QUEUE.put({"cn": text, "en": clean_res})
    except: pass

# --- 6. ALIYUN CALLS (CONTEXT-FREE) ---
def on_result_changed(message, *args):
    try:
        res = json.loads(message)['payload']['result']
        IF_CN_QUEUE.put(res) # Push to GLOBAL queue
    except: pass

def on_sentence_end(message, *args):
    try:
        res = json.loads(message)['payload']['result']
        threading.Thread(target=translate_task, args=(res,), daemon=True).start()
    except: pass

# --- 7. UI & PROCESSING ---
st.title("🪷 TBS Pro Translator")

tab_voice, tab_manual = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab_voice:
    active_key = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
    webrtc_ctx = webrtc_streamer(
        key="asr", mode=WebRtcMode.SENDONLY, 
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True, audio_receiver_size=1024
    )

with tab_manual:
    m_in = st.text_area("Type Chinese:", height=100)
    if st.button("Translate Now"):
        threading.Thread(target=translate_task, args=(m_in,), daemon=True).start()

# --- 8. THE MAIN LOOP (Drains Global Queues) ---
if webrtc_ctx.state.playing:
    if "sr" not in st.session_state:
        token = get_aliyun_token()
        st.session_state.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=active_key,
            on_result_changed=on_result_changed,
            on_sentence_end=on_sentence_end
        )
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    if webrtc_ctx.audio_receiver:
        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
            for frame in frames:
                for r_frame in st.session_state.resampler.resample(frame):
                    st.session_state.sr.send_audio(r_frame.to_ndarray().tobytes())
        except: pass

    # SAFE UI UPDATES: Pick up from Global Queues
    while not IF_CN_QUEUE.empty():
        st.session_state.live_cn = IF_CN_QUEUE.get()
    
    while not IF_EN_QUEUE.empty():
        entry = IF_EN_QUEUE.get()
        st.session_state.history.append(entry)
        st.session_state.live_cn = ""

    time.sleep(0.1)
    st.rerun()

elif "sr" in st.session_state:
    try: st.session_state.sr.stop()
    except: pass
    del st.session_state.sr

st.divider()

# --- 9. DISPLAY ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='panel-header'>Chinese Source</div>", unsafe_allow_html=True)
    src_h = "<div class='translation-card'>"
    if st.session_state.live_cn:
        src_h += f"<div class='text-display live-stream'>{st.session_state.live_cn}</div>"
    for i in reversed(st.session_state.history):
        src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='panel-header'>English Translation</div>", unsafe_allow_html=True)
    tar_h = "<div class='translation-card'>"
    for i in reversed(st.session_state.history):
        st.markdown(f"<div class='target-display'>{i['en']}</div>", unsafe_allow_html=True)
        st.code(i['en'], language="text") # Google Copy Icon
        st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain Status: {'Online' if vector_store else 'Offline'}")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
