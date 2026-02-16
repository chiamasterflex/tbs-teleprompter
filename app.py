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

# --- 1. CONFIGURATION ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 2. CSS STYLING (Google Style) ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 24px; height: 55vh; overflow-y: auto; }
    .text-display { font-size: 24px; color: #3c4043; line-height: 1.6; margin-bottom: 20px; }
    .target-display { font-size: 24px; color: #1a73e8; line-height: 1.6; font-weight: 400; margin-bottom: 20px; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 12px; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #70757a; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. PERSISTENT STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
if 'live_en' not in st.session_state: st.session_state.live_en = ""
if 'dialect' not in st.session_state: st.session_state.dialect = "Mandarin"

# Thread-safe queues to bridge Alibaba/DeepSeek to Streamlit
if 'cn_queue' not in st.session_state: st.session_state.cn_queue = queue.Queue()
if 'en_queue' not in st.session_state: st.session_state.en_queue = queue.Queue()

@st.cache_resource
def get_brain():
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if os.path.exists(DB_PATH):
            return FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)
    except: return None
    return None

vector_store = get_brain()

# --- 4. ENGINE FUNCTIONS ---
def get_aliyun_token():
    client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
    req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
    req.set_version('2019-07-17'); req.set_action_name('CreateToken')
    return json.loads(client.do_action_with_exception(req))['Token']['Id']

def translate_worker_logic(text):
    """Heavy lifting for translation, runs in background and pushes to en_queue"""
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
        # Filter out any leftover Chinese characters
        clean_res = re.sub(r'[\u4e00-\u9fff]+', '', res_text)
        st.session_state.en_queue.put({"cn": text, "en": clean_res})
    except: pass

# --- 5. ALIBABA INTEGRATION ---
def start_aliyun_session(appkey):
    if "sr" not in st.session_state:
        token = get_aliyun_token()
        
        def on_sentence_end(message, *args):
            txt = json.loads(message)['payload']['result']
            # Don't update UI here (avoids ScriptRunContext error)
            # Just push to translation logic
            threading.Thread(target=translate_worker_logic, args=(txt,), daemon=True).start()

        st.session_state.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: st.session_state.cn_queue.put(json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end
        )
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

# --- 6. UI ---
st.title("🪷 TBS Pro Translator")

tab_v, tab_m = st.tabs(["🎙️ Voice Mode", "⌨️ Manual Mode"])

with tab_v:
    active_key = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
    webrtc_ctx = webrtc_streamer(
        key="asr", mode=WebRtcMode.SENDONLY, 
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True, audio_receiver_size=1024
    )

with tab_m:
    m_in = st.text_area("Type Chinese:", height=100)
    if st.button("Translate Now"):
        threading.Thread(target=translate_worker_logic, args=(m_in,), daemon=True).start()

# --- 7. THE CONTROLLER (Context Safe) ---
if webrtc_ctx.state.playing:
    start_aliyun_session(active_key)
    if webrtc_ctx.audio_receiver:
        try:
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
            for frame in frames:
                for r_frame in st.session_state.resampler.resample(frame):
                    st.session_state.sr.send_audio(r_frame.to_ndarray().tobytes())
        except: pass

    # DRAIN QUEUES: This is where we safely update the UI
    while not st.session_state.cn_queue.empty():
        st.session_state.live_cn = st.session_state.cn_queue.get()
    
    while not st.session_state.en_queue.empty():
        new_entry = st.session_state.en_queue.get()
        st.session_state.history.append(new_entry)
        st.session_state.live_cn = "" # Clear live when sentence ends

    time.sleep(0.1)
    st.rerun()

elif "sr" in st.session_state:
    try: st.session_state.sr.stop()
    except: pass
    del st.session_state.sr

st.divider()

# --- 8. DISPLAY ---
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
        # Result text
        st.markdown(f"<div class='target-display'>{i['en']}</div>", unsafe_allow_html=True)
        # Copy Icon (st.code provides the icon naturally)
        st.code(i['en'], language="text")
        st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain Status: {'Online' if vector_store else 'Offline'}")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
