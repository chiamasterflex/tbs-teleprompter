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

# --- 1. CONFIG & CLIENTS ---
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
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.5; margin-bottom: 15px; }
    .target-display { font-size: 22px; color: #1a73e8; line-height: 1.5; font-weight: 500; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. PERSISTENT STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'live_cn' not in st.session_state: st.session_state.live_cn = ""
if 'live_en' not in st.session_state: st.session_state.live_en = ""
if 'dialect' not in st.session_state: st.session_state.dialect = "Mandarin"
if 'last_latency' not in st.session_state: st.session_state.last_latency = 0.0

@st.cache_resource
def get_brain():
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)
    return None

vector_store = get_brain()

# --- 4. THE ENGINE (THREAD-SAFE) ---
def get_aliyun_token():
    client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
    req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
    req.set_version('2019-07-17'); req.set_action_name('CreateToken')
    return json.loads(client.do_action_with_exception(req))['Token']['Id']

def translate_text(text, is_final=True):
    start_t = time.time()
    prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
    if vector_store:
        docs = vector_store.similarity_search(text, k=2)
        prompt += "\nContext: " + " ".join([d.page_content for d in docs])
    
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
            if not is_final: st.session_state.live_en = full_res
            
    if is_final:
        st.session_state.last_latency = round(time.time() - start_t, 2)
        st.session_state.history.append({"cn": text, "en": full_res.strip(), "lat": st.session_state.last_latency})
        st.session_state.live_en = ""

# --- 5. AUDIO CALLBACK ---
# This function runs INSIDE the WebRTC component, preventing the 'NoneType' crash
def audio_frame_callback(frame: av.AudioFrame):
    # This part is complex but crucial: we only initialize the ASR when audio actually flows
    if "sr" not in st.session_state:
        appkey = ALIYUN_APP_MANDARIN if st.session_state.dialect == "Mandarin" else ALIYUN_APP_CANTONESE
        token = get_aliyun_token()
        
        def on_sentence_end(message, *args):
            res = json.loads(message)['payload']['result']
            translate_text(res, is_final=True)
            st.session_state.live_cn = ""

        st.session_state.sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(st.session_state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end
        )
        st.session_state.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        st.session_state.sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})

    # Resample and send audio to Alibaba
    for r_frame in st.session_state.resampler.resample(frame):
        st.session_state.sr.send_audio(r_frame.to_ndarray().tobytes())
    return frame

# --- 6. UI ---
st.title("🪷 TBS Pro Translator")

tab1, tab2 = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab1:
    webrtc_ctx = webrtc_streamer(
        key="asr",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback, # CRITICAL FIX
    )

with tab2:
    m_in = st.text_area("Type Chinese:", placeholder="Instant translation coming soon...")
    if st.button("Translate"): translate_text(m_in)

# CLEANUP LOGIC: Stops rogue threads
if not webrtc_ctx.state.playing and "sr" in st.session_state:
    st.session_state.sr.stop()
    del st.session_state.sr

st.divider()

# --- DISPLAY ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Chinese Source")
    src_h = "<div class='translation-card'>"
    if st.session_state.live_cn: src_h += f"<div class='text-display live-stream'>{st.session_state.live_cn}</div>"
    for i in reversed(st.session_state.history): src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", 1)

with c2:
    st.markdown("### English Translation")
    tar_h = "<div class='translation-card'>"
    if st.session_state.live_en: tar_h += f"<div class='target-display live-stream'>{st.session_state.live_en}</div>"
    for i in reversed(st.session_state.history):
        st.markdown(f"<div class='target-display'>{i['en']}</div>", 1)
        st.code(i['en'], language="text")
        st.caption(f"Latency: {i['lat']}s")
        st.divider()
    st.markdown("</div>", 1)

# RAG Check
st.sidebar.write(f"🧠 Brain: {vector_store.index.ntotal if vector_store else 0} items")
if st.sidebar.button("Clear History"): st.session_state.history = []; st.rerun()

# This keeps the UI updating as text streams in
if webrtc_ctx.state.playing:
    time.sleep(0.1)
    st.rerun()
