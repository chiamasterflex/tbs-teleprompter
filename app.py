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

st.set_page_config(page_title="TBS Pro Translator", layout="wide", initial_sidebar_state="expanded")

# --- 2. GOOGLE UI CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif !important; }
    
    .translation-card {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 24px;
        min-height: 400px;
        height: 50vh;
        overflow-y: auto;
        box-shadow: 0 1px 3px rgba(60,64,67,.3);
    }
    
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.6; margin-bottom: 20px; }
    .target-display { font-size: 22px; color: #1a73e8; line-height: 1.6; font-weight: 400; margin-bottom: 20px; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 10px; }
    .panel-header { font-size: 13px; font-weight: 500; color: #70757a; text-transform: uppercase; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. RAG CORE ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def generate_dynamic_prompt(chinese_text, vector_store):
    prompt = "You are the official TBS translator. Rules: 1. 100% English. 2. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master. Output ONLY translation."
    if vector_store:
        try:
            docs = vector_store.similarity_search(chinese_text, k=3)
            if docs:
                prompt += "\nRef: " + " ".join([d.page_content for d in docs])
        except: pass
    return prompt

# --- 4. STATE ---
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

# --- 5. WORKER ---
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
                temp_res = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        temp_res += content
                        if not is_final: app_state['live_en'] = temp_res
                if is_final:
                    clean_res = re.sub(r'[\u4e00-\u9fff]+', '', temp_res).strip()
                    lat = round(time.time() - start_t, 2)
                    app_state['last_latency'] = lat
                    app_state['history'].append({"cn": text, "en": clean_res, "lat": lat})
                    app_state['live_en'] = ""
            except: pass
            finally: q.task_done()
    threading.Thread(target=translation_worker, args=(st.session_state['trans_queue'], state), daemon=True).start()

# --- 6. VOICE ENGINE ---
def start_aliyun(state, webrtc_ctx, q, appkey):
    try:
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        req.set_version('2019-07-17'); req.set_action_name('CreateToken')
        token = json.loads(client.do_action_with_exception(req))['Token']['Id']
        
        sr = nls.NlsSpeechTranscriber(url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=lambda m, *a: q.put((json.loads(m)['payload']['result'], True, state['vector_store'])))
        
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})
        
        while state['run'] and webrtc_ctx.state.playing:
            if webrtc_ctx.audio_receiver:
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                for frame in frames:
                    for r_frame in resampler.resample(frame):
                        sr.send_audio(r_frame.to_ndarray().tobytes())
        sr.stop()
    except: pass

# --- 7. MAIN UI ---
st.title("🪷 TBS Pro Translator")

tab_voice, tab_manual = st.tabs(["🎙️ Voice", "⌨️ Manual"])

with tab_voice:
    active_key = ALIYUN_APPKEY_MANDARIN if state['dialect'] == "Mandarin" else ALIYUN_APPKEY_CANTONESE
    webrtc_ctx = webrtc_streamer(key="asr", mode=WebRtcMode.SENDONLY, 
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_receiver_size=4096)
    if webrtc_ctx.state.playing and not state['run']:
        state['run'] = True
        threading.Thread(target=start_aliyun, args=(state, webrtc_ctx, st.session_state['trans_queue'], active_key), daemon=True).start()
    elif not webrtc_ctx.state.playing and state['run']: state['run'] = False

with tab_manual:
    m_input = st.text_area("Type or Paste Chinese", height=150, key="manual_area")
    # Trigger instantly if text changed
    if m_input and (not state['history'] or m_input != state['history'][-1]['cn']):
        if st.button("Instant Translate"): # Keep button as a 'Finalize' trigger but logic is active
            st.session_state['trans_queue'].put((m_input, True, state['vector_store']))

# --- STATUS BAR ---
brain_count = state['vector_store'].index.ntotal if state['vector_store'] else 0
st.caption(f"Latency: {state['last_latency']}s | 🧠 Brain: {brain_count} items")

st.divider()

# --- DISPLAY PANELS ---
col_src, col_tar = st.columns(2)
with col_src:
    st.markdown("<div class='panel-header'>Source</div>", 1)
    src_h = "<div class='translation-card'>"
    if state['live_cn']: src_h += f"<div class='text-display live-stream'>{state['live_cn']}</div>"
    for i in reversed(state['history']): src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", 1)

with col_tar:
    st.markdown("<div class='panel-header'>Translation</div>", 1)
    tar_h = "<div class='translation-card'>"
    if state['live_en']: tar_h += f"<div class='target-display live-stream'>{state['live_en']}</div>"
    for i in reversed(state['history']):
        st.markdown(f"<div class='target-display'>{i['en']}</div>", 1)
        st.code(i['en'], language="text")
        st.divider()
    st.markdown("</div>", 1)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    state['dialect'] = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    if st.button("Clear"): state['history'] = []; st.rerun()
    st.divider()
    up = st.file_uploader("Train Brain", type=['pdf','txt','csv'], accept_multiple_files=True)
    if st.button("Learn"):
        if up:
            with st.spinner("Learning..."):
                docs = []
                for f in up:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(f.getvalue()); p = tmp.name
                    ldr = PyPDFLoader(p) if f.name.endswith('.pdf') else TextLoader(p)
                    docs.extend(ldr.load()); os.remove(p)
                chnk = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)
                if state['vector_store'] is None: state['vector_store'] = FAISS.from_documents(chnk, get_embedding_model())
                else: state['vector_store'].add_documents(chnk)
                state['vector_store'].save_local(DB_PATH); st.rerun()

if state['run']: time.sleep(0.4); st.rerun()
