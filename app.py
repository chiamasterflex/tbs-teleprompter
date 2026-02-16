import streamlit as st
import time
import queue
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

# --- 1. GLOBAL QUEUES (Thread-Safe, Context-Independent) ---
if "CN_LIVE_Q" not in st.session_state:
    st.session_state.CN_LIVE_Q = queue.Queue()
if "TRANSLATE_Q" not in st.session_state:
    st.session_state.TRANSLATE_Q = queue.Queue()
if "HISTORY_Q" not in st.session_state:
    st.session_state.HISTORY_Q = queue.Queue()

# --- 2. CONFIG & SECRETS (Fetched ONCE in main thread) ---
AK_ID = st.secrets["ALIYUN_AK_ID"]
AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DS_KEY = st.secrets["DEEPSEEK_API_KEY"]
APP_MANDARIN = "kNPPGUGiUNqBa7DB"
APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 3. CSS (Google Translate Style) ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 24px; height: 52vh; overflow-y: auto; }
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.6; margin-bottom: 20px; }
    .target-display { font-size: 24px; color: #1a73e8; line-height: 1.6; font-weight: 400; margin-bottom: 20px; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 4px solid #e67e22; padding-left: 12px; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #70757a; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 4. DATA & RAG ---
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

# --- 5. TRANSLATION WORKER ---
def translation_job(text, v_store, hist_q):
    """Runs in a background thread, pushes result to HISTORY_Q"""
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if v_store:
        docs = v_store.similarity_search(text, k=2)
        prompt += "\nRef: " + " ".join([d.page_content for d in docs])
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.1
        )
        res = response.choices[0].message.content.strip()
        hist_q.put({"cn": text, "en": re.sub(r'[\u4e00-\u9fff]+', '', res)})
    except: pass

# --- 6. AUDIO CALLBACK (THREAD-SAFE) ---
def audio_frame_callback(frame: av.AudioFrame):
    # NEVER touch st.session_state or st.secrets here. Use local variables.
    if not hasattr(audio_frame_callback, "sr"):
        # Initialize Alibaba using variables passed from the main thread
        client = AcsClient(AK_ID, AK_SECRET, "ap-southeast-1")
        req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        req.set_version('2019-07-17'); req.set_action_name('CreateToken')
        token = json.loads(client.do_action_with_exception(req))['Token']['Id']
        appkey = APP_MANDARIN if st.session_state.dialect == "Mandarin" else APP_CANTONESE

        audio_frame_callback.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: st.session_state.CN_LIVE_Q.put(json.loads(m)['payload']['result']),
            on_sentence_end=lambda m, *a: st.session_state.TRANSLATE_Q.put(json.loads(m)['payload']['result'])
        )
        audio_frame_callback.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        audio_frame_callback.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    for r_frame in audio_frame_callback.resampler.resample(frame):
        audio_frame_callback.sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 7. UI ---
st.title("🪷 TBS Pro Translator")

tab_v, tab_m = st.tabs(["🎙️ Voice Mode", "⌨️ Manual Mode"])

with tab_v:
    webrtc_ctx = webrtc_streamer(
        key="asr",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback,
        async_processing=True,
        audio_receiver_size=4096
    )

with tab_m:
    m_in = st.text_area("Manual Input (Chinese):", height=100)
    if st.button("Translate Text"):
        threading.Thread(target=translation_job, args=(m_in, vector_store, st.session_state.HISTORY_Q)).start()

# --- 8. THE CONTROLLER (Drains Queues & Updates UI) ---
if webrtc_ctx.state.playing:
    # 1. Update Live CN
    while not st.session_state.CN_LIVE_Q.empty():
        st.session_state.live_cn = st.session_state.CN_LIVE_Q.get()
    
    # 2. Trigger Translations
    while not st.session_state.TRANSLATE_Q.empty():
        final_txt = st.session_state.TRANSLATE_Q.get()
        threading.Thread(target=translation_job, args=(final_txt, vector_store, st.session_state.HISTORY_Q)).start()
    
    # 3. Collect History
    while not st.session_state.HISTORY_Q.empty():
        st.session_state.history.append(st.session_state.HISTORY_Q.get())
        st.session_state.live_cn = ""

    time.sleep(0.1)
    st.rerun()

elif hasattr(audio_frame_callback, "sr"):
    try: audio_frame_callback.sr.stop()
    except: pass
    del audio_frame_callback.sr

st.divider()

# --- 9. DISPLAY PANELS ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='panel-header'>Source</div>", 1)
    src_h = "<div class='translation-card'>"
    if st.session_state.live_cn:
        src_h += f"<div class='text-display live-stream'>{st.session_state.live_cn}</div>"
    for i in reversed(st.session_state.history):
        src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='panel-header'>Translation</div>", 1)
    tar_h = "<div class='translation-card'>"
    for i in reversed(st.session_state.history):
        # Text result
        st.markdown(f"<div class='target-display'>{i['en']}</div>", unsafe_allow_html=True)
        # Copy Icon (Google Style)
        st.code(i['en'], language="text")
        st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.session_state.dialect = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    st.write(f"🧠 Brain: {vector_store.index.ntotal if vector_store else 0} fragments")
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.rerun()
