import streamlit as st
import time
import threading
import queue
import re
import os
import tempfile
import shutil
import json
import traceback

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

# --- 1. CONFIGURATION (ST.SECRETS FOR ONLINE) ---
try:
    ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
    ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except KeyError:
    st.error("Please set ALIYUN_AK_ID, ALIYUN_AK_SECRET, and DEEPSEEK_API_KEY in Streamlit Secrets.")
    st.stop()

ALIYUN_APPKEY_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APPKEY_CANTONESE = "B63clvkhBpyeehDk"

DB_PATH = "tbs_knowledge_db"
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Teleprompter", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Open Sans', sans-serif !important; }
    .scroll-container { 
        height: 48vh; 
        min-height: 400px; 
        overflow-y: auto; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        padding: 20px; 
        background-color: #ffffff; 
        margin-bottom: 20px; 
        display: flex;
        flex-direction: column;
    }
    .committed-cn, .committed-en { font-size: 19px; color: #31333F; line-height: 1.6; font-weight: 400; margin-bottom: 10px; }
    .live-cn { font-size: 20px; color: #1565C0; font-weight: 600; line-height: 1.6; margin-bottom: 15px; border-left: 3px solid #1565C0; padding-left: 10px; }
    .live-en { font-size: 20px; color: #E65100; font-weight: 600; line-height: 1.6; font-style: italic; margin-bottom: 15px; border-left: 3px solid #E65100; padding-left: 10px; }
    .sep { border: 0; border-top: 1px solid #f0f2f6; margin: 15px 0; }
    .panel-header { font-size: 13px; font-weight: 600; text-transform: uppercase; color: #757575; margin-bottom: 5px; }
    .latency-tag { font-size: 11px; color: #9e9e9e; margin-top: -5px; margin-bottom: 10px; }
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
                prompt += "\nRef terms: " + " ".join([d.page_content for d in docs])
        except: pass
    return prompt

# --- 4. STATE & WORKER ---
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
                full_res = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_res += content
                        if not is_final: app_state['live_en'] = full_res + "..."
                if is_final:
                    clean_res = re.sub(r'[\u4e00-\u9fff]+', '', full_res).strip()
                    lat = round(time.time() - start_t, 2)
                    app_state['history'].append({"cn": text, "en": clean_res, "lat": lat})
                    app_state['live_en'] = ""
                    app_state['last_latency'] = lat
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

# --- 6. SIDEBAR & ADMIN ---
with st.sidebar:
    st.header("⚙️ Settings")
    state['dialect'] = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    if st.button("🧹 Clear History"):
        state['history'] = []
        st.rerun()
    st.divider()
    st.subheader("🧠 Brain Status")
    brain_count = state['vector_store'].index.ntotal if state['vector_store'] else 0
    st.info(f"Loaded fragments: {brain_count}")
    
    up = st.file_uploader("Upload Teachings", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)
    if st.button("➕ Train"):
        if up:
            with st.spinner("Learning..."):
                docs = []
                for uf in up:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uf.getvalue()); tmp_p = tmp.name
                    ldr = PyPDFLoader(tmp_p) if uf.name.endswith('.pdf') else TextLoader(tmp_p)
                    docs.extend(ldr.load()); os.remove(tmp_p)
                chnk = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)
                if state['vector_store'] is None: state['vector_store'] = FAISS.from_documents(chnk, get_embedding_model())
                else: state['vector_store'].add_documents(chnk)
                state['vector_store'].save_local(DB_PATH); st.rerun()

# --- 7. MAIN UI TABS ---
st.title("🪷 TBS Pro Translator")

tab_voice, tab_manual = st.tabs(["🎙️ Voice Translation", "⌨️ Manual Text"])

with tab_voice:
    c1, c2 = st.columns([1, 2])
    with c1:
        active_key = ALIYUN_APPKEY_MANDARIN if state['dialect'] == "Mandarin" else ALIYUN_APPKEY_CANTONESE
        webrtc_ctx = webrtc_streamer(
            key="asr", mode=WebRtcMode.SENDONLY, 
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
            audio_receiver_size=4096 # BUFFER FIX
        )
    with c2:
        st.caption(f"Status: {state['status']} | Latency: {state['last_latency']}s | 🧠 RAG: {brain_count}")

with tab_manual:
    m_input = st.text_area("Chinese Input:", placeholder="Paste text and press Ctrl+Enter...")
    if m_input and (not state['history'] or m_input != state['history'][-1]['cn']):
        # Instant trigger on content change
        st.session_state['trans_queue'].put((m_input, True, state['vector_store']))

if webrtc_ctx.state.playing and not state['run']:
    state['run'] = True
    threading.Thread(target=start_aliyun, args=(state, webrtc_ctx, st.session_state['trans_queue'], active_key), daemon=True).start()
elif not webrtc_ctx.state.playing and state['run']:
    state['run'] = False

st.divider()

# --- 8. BILINGUAL DISPLAY PANELS ---
col_cn, col_en = st.columns(2)

with col_cn:
    st.markdown("<div class='panel-header'>Chinese Source</div>", unsafe_allow_html=True)
    with st.container(border=True, height=500):
        if state['live_cn']:
            st.markdown(f"<div class='live-cn'>{state['live_cn']}</div>", unsafe_allow_html=True)
        for i in reversed(state['history']):
            st.markdown(f"<div class='committed-cn'>{i['cn']}</div>", unsafe_allow_html=True)
            st.divider()

with col_en:
    st.markdown("<div class='panel-header'>English Translation</div>", unsafe_allow_html=True)
    with st.container(border=True, height=500):
        if state['live_en']:
            st.markdown(f"<div class='live-en'>{state['live_en']}</div>", unsafe_allow_html=True)
        for i in reversed(state['history']):
            st.markdown(f"<div class='committed-en'>{i['en']}</div>", unsafe_allow_html=True)
            st.code(i['en'], language="text") # COPY ICON FIX
            st.markdown(f"<div class='latency-tag'>Lat: {i['lat']}s</div>", unsafe_allow_html=True)
            st.divider()

if state['run']: 
    time.sleep(0.4)
    st.rerun()
