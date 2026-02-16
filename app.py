import streamlit as st
import time
import threading
import queue
import re
import os
import tempfile
import shutil
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

st.set_page_config(page_title="TBS Pro Translator", layout="wide", initial_sidebar_state="collapsed")

# --- 2. GOOGLE TRANSLATE STYLE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif !important; }
    
    .main-card {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 20px;
        min-height: 400px;
        height: 55vh;
        overflow-y: auto;
    }
    
    .text-entry {
        padding: 15px;
        border-bottom: 1px solid #e8eaed;
        display: flex;
        flex-direction: column;
    }
    
    .source-text { font-size: 20px; color: #3c4043; line-height: 1.5; }
    .target-text { font-size: 20px; color: #1a73e8; line-height: 1.5; font-weight: 500; }
    .live-target { color: #e67e22; font-style: italic; }
    
    .panel-header {
        font-size: 13px;
        font-weight: 500;
        color: #70757a;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
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
                prompt += "\nRef: " + " ".join([d.page_content for d in docs])
        except: pass
    return prompt

# --- 4. STATE INITIALIZATION ---
if 'app_state' not in st.session_state:
    st.session_state['app_state'] = {
        'history': [], 'live_cn': "", 'live_en': "", 'run': False,
        'vector_store': None, 'status': "Idle", 'dialect': "Mandarin", 'last_latency': 0.0
    }
state = st.session_state['app_state']

if state['vector_store'] is None and os.path.exists(DB_PATH):
    try:
        state['vector_store'] = FAISS.load_local(DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    except: pass

if 'trans_queue' not in st.session_state:
    st.session_state['trans_queue'] = queue.Queue()
    def translation_worker(q, app_state):
        while True:
            item = q.get()
            if item is None: break
            text, is_final, v_store = item
            start_t = time.time()
            try:
                prompt = generate_dynamic_prompt(text, v_store)
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                    stream=True, temperature=0.1
                )
                temp_full_res = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        temp_full_res += content
                        if not is_final: app_state['live_en'] = temp_full_res
                if is_final:
                    clean_res = re.sub(r'[\u4e00-\u9fff]+', '', temp_full_res).strip()
                    latency = round(time.time() - start_t, 2)
                    app_state['last_latency'] = latency
                    app_state['history'].append({"cn": text, "en": clean_res, "lat": latency})
                    app_state['live_en'] = ""
            except: pass
            finally: q.task_done()
    threading.Thread(target=translation_worker, args=(st.session_state['trans_queue'], state), daemon=True).start()

# --- 5. ALIYUN ENGINE ---
def start_aliyun(state, webrtc_ctx, q, appkey):
    token, _ = get_aliyun_token()
    if not token: return
    def on_result_changed(message, *args):
        text = json.loads(message)['payload']['result']
        state['live_cn'] = text
        if len(text) > 10 and len(text) % 8 == 0: q.put((text, False, state['vector_store']))
    def on_sentence_end(message, *args):
        text = json.loads(message)['payload']['result']
        clean_text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text).strip()
        if len(clean_text) >= 2: q.put((clean_text, True, state['vector_store']))
        state['live_cn'] = ""
    sr = nls.NlsSpeechTranscriber(url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
                                  token=token, appkey=appkey,
                                  on_result_changed=on_result_changed, on_sentence_end=on_sentence_end)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})
    while state['run'] and webrtc_ctx.state.playing:
        if webrtc_ctx.audio_receiver:
            try:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                for frame in frames:
                    for r_frame in resampler.resample(frame):
                        sr.send_audio(r_frame.to_ndarray().tobytes())
            except: break
    sr.stop()

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⌨️ Manual Input")
    manual_input = st.text_area("Chinese Text:", placeholder="Paste and click translate...")
    if st.button("Translate Now"):
        if manual_input:
            st.session_state['trans_queue'].put((manual_input, True, state['vector_store']))

    st.divider()
    state['dialect'] = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    if st.button("🧹 Clear History"):
        state['history'] = []
        state['last_latency'] = 0.0
        st.rerun()

# --- 7. MAIN UI ---
st.title("🪷 TBS Pro Translator")

webrtc_ctx = webrtc_streamer(
    key="asr", mode=WebRtcMode.SENDONLY, 
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True, audio_receiver_size=1024 
)

brain_count = state['vector_store'].index.ntotal if state['vector_store'] else 0
st.caption(f"Status: {state['status']} | Latency: {state['last_latency']}s | 🧠 Brain: {brain_count} items")

if webrtc_ctx.state.playing and not state['run']:
    state['run'] = True
    active_key = ALIYUN_APPKEY_MANDARIN if state['dialect'] == "Mandarin" else ALIYUN_APPKEY_CANTONESE
    threading.Thread(target=start_aliyun, args=(state, webrtc_ctx, st.session_state['trans_queue'], active_key), daemon=True).start()
    st.rerun()
elif not webrtc_ctx.state.playing and state['run']:
    state['run'] = False
    st.rerun()

st.divider()

# DISPLAY PANELS
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='panel-header'>Chinese Source</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if state['live_cn']:
            st.markdown(f"<div class='source-text'>{state['live_cn']}</div>", unsafe_allow_html=True)
        for i in reversed(state['history']):
            st.markdown(f"<div class='source-text'>{i['cn']}</div>", unsafe_allow_html=True)
            st.divider()

with c2:
    st.markdown("<div class='panel-header'>English Translation</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if state['live_en']:
            st.markdown(f"<div class='target-text live-target'>{state['live_en']}</div>", unsafe_allow_html=True)
        for i in reversed(state['history']):
            # Using st.code here provides a built-in COPY button on hover!
            st.markdown(f"<div class='target-text'>{i['en']}</div>", unsafe_allow_html=True)
            st.code(i['en'], language="text") # This is your Copy Icon solution
            st.caption(f"Latency: {i['lat']}s")
            st.divider()

if state['run']: time.sleep(0.4); st.rerun()
