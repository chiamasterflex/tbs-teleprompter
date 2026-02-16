import streamlit as st
import time
import threading
import queue
import re
import os
import tempfile
import json

# --- WEB AUDIO LIBRARIES ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- ALIBABA & OPENAI ---
import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from openai import OpenAI

# --- RAG ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

ALIYUN_APPKEY_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APPKEY_CANTONESE = "B63clvkhBpyeehDk"

DB_PATH = "tbs_knowledge_db"
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Open Sans', sans-serif !important; }
    .scroll-container { display: flex; flex-direction: column-reverse; height: 50vh; min-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; background-color: #ffffff; }
    .committed-cn, .committed-en { font-size: 20px; color: #31333F; line-height: 1.6; }
    .live-cn { font-size: 22px; color: #1565C0; font-weight: 600; line-height: 1.6; border-left: 4px solid #1565C0; padding-left: 12px; }
    .live-en { font-size: 22px; color: #E65100; font-weight: 600; line-height: 1.6; font-style: italic; border-left: 4px solid #E65100; padding-left: 12px; }
    .sep { border: 0; border-top: 1px solid #f0f2f6; margin: 15px 0; }
    .panel-header { font-size: 14px; font-weight: 600; text-transform: uppercase; color: #757575; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def get_aliyun_token():
    try:
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        request.set_version('2019-07-17')
        request.set_action_name('CreateToken')
        response_dict = json.loads(client.do_action_with_exception(request))
        return response_dict['Token']['Id'], None
    except Exception as e: return None, str(e)

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def generate_dynamic_prompt(chinese_text, vector_store):
    prompt = "You are the official TBS translator. Rules: 1. 100% English. 2. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master. Output ONLY translation."
    if vector_store:
        try:
            docs = vector_store.similarity_search(chinese_text, k=2)
            if docs:
                prompt += "\nRef Knowledge: " + " ".join([d.page_content for d in docs])
        except: pass
    return prompt

# --- 4. STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
if 'live_en' not in st.session_state: st.session_state.live_en = ""
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'dialect' not in st.session_state: st.session_state.dialect = "Mandarin"

if st.session_state.vector_store is None and os.path.exists(DB_PATH):
    try:
        st.session_state.vector_store = FAISS.load_local(DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    except: pass

# --- 5. THE TRANSLATION ENGINE ---
def run_translation(text, is_final):
    try:
        prompt = generate_dynamic_prompt(text, st.session_state.vector_store)
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
            clean_res = re.sub(r'[\u4e00-\u9fff]+', '', full_res).strip()
            st.session_state.history.append({"cn": text, "en": clean_res})
            st.session_state.live_en = ""
    except: pass

# --- 6. ALIYUN ENGINE ---
def audio_frame_callback(frame: av.AudioFrame):
    # This is much more stable than threading for high receiver sizes
    if "sr" not in st.session_state:
        token, _ = get_aliyun_token()
        appkey = ALIYUN_APPKEY_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APPKEY_CANTONESE
        
        def on_sentence_end(message, *args):
            text = json.loads(message)['payload']['result']
            run_translation(text, True)
            st.session_state.live_cn = ""

        st.session_state.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(st.session_state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end
        )
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    for r_frame in st.session_state.resampler.resample(frame):
        st.session_state.sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 7. MAIN UI ---
st.title("🪷 TBS Pro Translator")

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.dialect = st.selectbox("Speech Dialect:", ["Mandarin", "Cantonese"])
    if st.button("🧹 Clear History"):
        st.session_state.history = []; st.rerun()
    st.divider()
    brain_count = st.session_state.vector_store.index.ntotal if st.session_state.vector_store else 0
    st.info(f"🧠 Brain: {brain_count} items")

# Manual Input Tab
tab_voice, tab_manual = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab_voice:
    webrtc_ctx = webrtc_streamer(
        key="asr",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback, # Using callback for stability
    )

with tab_manual:
    m_input = st.text_area("Type Chinese:")
    if st.button("Translate Text"):
        run_translation(m_input, True)

# Cleanup: If mic is off, stop the transcriber
if webrtc_ctx and not webrtc_ctx.state.playing and "sr" in st.session_state:
    st.session_state.sr.stop()
    del st.session_state.sr

st.divider()

# --- DISPLAY PANELS ---
def get_html(k):
    html = f"<div class='scroll-container'>"
    if st.session_state[f'live_{k}']:
        html += f"<div class='live-{k}'>{st.session_state[f'live_{k}']}</div>"
    for i in reversed(st.session_state.history):
        html += f"<div class='committed-{k}'>{i[k]}</div><hr class='sep'>"
    return html + "</div>"

c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='panel-header'>Chinese Source</div>", unsafe_allow_html=True)
    st.markdown(get_html("cn"), unsafe_allow_html=True)
with c2:
    st.markdown("<div class='panel-header'>English Translation</div>", unsafe_allow_html=True)
    st.markdown(get_html("en"), unsafe_allow_html=True)

# Heartbeat UI refresh
if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.4)
    st.rerun()
