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
import logging

# --- NEW WEB AUDIO LIBRARIES ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- MAC SSL BYPASS ---
import ssl
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['WEBSOCKET_CLIENT_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Alibaba Cloud NLS (WebSocket) & Core SDK
import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

# DeepSeek 
from openai import OpenAI

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION ---
ALIYUN_AK_ID = "LTAI5tJKHDFcaiEGrCDdtAJy"
ALIYUN_AK_SECRET = "8ca7iMVmV9iQYg8QGlxLTWajlg7yCy"

# DIALECT APPKEYS
ALIYUN_APPKEY_MANDARIN = "kNPPGUGiUNqBa7DB" 
ALIYUN_APPKEY_CANTONESE = "B63clvkhBpyeehDk" 

ALIYUN_VOCAB_ID = "" 
ALIYUN_CUSTOM_MODEL_ID = "d6e3d70e9230455384e225521adcc1f6"

DEEPSEEK_API_KEY = "sk-43a90c6020b44e818013a112dd890b79" 
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Teleprompter", layout="wide", initial_sidebar_state="collapsed")

# --- 2. CSS STYLING (MOBILE RESPONSIVE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Open Sans', sans-serif !important; }
    
    /* Default Desktop View */
    .scroll-container { display: flex; flex-direction: column-reverse; height: 50vh; min-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; background-color: #ffffff; margin-bottom: 20px; }
    .committed-cn, .committed-en { font-size: 19px; color: #31333F; line-height: 1.6; font-weight: 400; }
    .live-cn { font-size: 19px; color: #1565C0; font-weight: 600; line-height: 1.6; }
    .live-en { font-size: 19px; color: #2E7D32; font-weight: 600; line-height: 1.6; }
    .sep { border: 0; border-top: 1px solid #f0f2f6; margin: 15px 0; }
    .panel-header { font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; color: #31333F; }
    .latency-tag { font-size: 12px; color: #888; font-style: italic; }
    
    /* Mobile Override View */
    @media (max-width: 768px) {
        .scroll-container { height: 32vh; min-height: 250px; padding: 15px; }
        .committed-cn, .committed-en { font-size: 16px; }
        .live-cn, .live-en { font-size: 16px; }
        .panel-header { font-size: 12px; margin-bottom: 5px; }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ALIYUN & RAG CORE ---
def get_aliyun_token():
    try:
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        request.set_version('2019-07-17')
        request.set_action_name('CreateToken')
        response_bytes = client.do_action_with_exception(request)
        response_dict = json.loads(response_bytes)
        if 'Token' in response_dict:
            return response_dict['Token']['Id'], None
        else:
            return None, f"Aliyun API Error: {response_dict.get('Message')}"
    except Exception as e: 
        return None, str(e)

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

TBS_BASE_PROMPT = """You are the official English translator for True Buddha School (TBS).
STRICT RULES:
1. Output MUST be 100% English. NO Chinese characters.
2. If stuck, use Pinyin. 
3. Terminology: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master, 蓮花童子=Padmakumara, 護摩=Homa.
Output ONLY the translation."""

def generate_dynamic_prompt(chinese_text, vector_store):
    dynamic_prompt = TBS_BASE_PROMPT
    if vector_store is not None:
        try:
            docs = vector_store.similarity_search(chinese_text, k=3)
            if docs:
                dynamic_prompt += "\n\n=== REFERENCE TEACHINGS ===\n"
                for i, doc in enumerate(docs): dynamic_prompt += f"- {doc.page_content}\n"
        except Exception: pass
    return dynamic_prompt

# --- 4. STATE & ASYNC WORKER ---
if 'app_state' not in st.session_state:
    st.session_state['app_state'] = {
        'history': [], 'live_cn': "", 'live_en': "", 'run': False,
        'vector_store': None, 'status': "Idle",
        'last_latency': 0, 'pending_count': 0, 'dialect': "Mandarin"
    }
state = st.session_state['app_state']

if state['vector_store'] is None and os.path.exists(DB_PATH):
    try:
        state['vector_store'] = FAISS.load_local(DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    except Exception: pass

if 'translation_queue' not in st.session_state:
    st.session_state['translation_queue'] = queue.Queue()
    
    def translation_worker(worker_queue):
        while True:
            item = worker_queue.get()
            if item is None: break
            text, v_store, app_state = item
            start_time = time.time()
            try:
                prompt = generate_dynamic_prompt(text, v_store)
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                    temperature=0.1
                )
                translation = response.choices[0].message.content
                translation = re.sub(r'[\u4e00-\u9fff]+', '', translation).strip()
                latency = round(time.time() - start_time, 2)
                app_state['history'].append({"cn": text, "en": translation, "latency": latency})
                app_state['last_latency'] = latency
            except: pass
            finally:
                app_state['pending_count'] = max(0, worker_queue.qsize())
                worker_queue.task_done()

    threading.Thread(target=translation_worker, args=(st.session_state['translation_queue'],), daemon=True).start()

# --- 5. ALIYUN WEBSOCKET ENGINE (WEBRTC ADAPTED) ---
def start_websocket_stream(state, webrtc_ctx, t_queue, selected_appkey):
    try:
        state['status'] = "🔄 Fetching Token..."
        token, error_msg = get_aliyun_token()
        if not token: 
            state['status'] = f"🔴 Token Error: {error_msg}"
            state['run'] = False
            return

        def on_sentence_begin(message, *args):
            state['status'] = f"🟢 Hearing audio ({state['dialect']})..."

        def on_result_changed(message, *args):
            try: 
                state['live_cn'] = json.loads(message)['payload']['result']
            except: pass

        def on_sentence_end(message, *args):
            try:
                result = json.loads(message)['payload']['result']
                clean_text = re.sub(r'[^\w\u4e00-\u9fff\s\.,!?]', '', result).strip()
                if len(clean_text) >= 2:
                    t_queue.put((clean_text, state['vector_store'], state))
                    state['pending_count'] = t_queue.qsize()
                state['live_cn'] = ""
            except: pass

        def on_error(message, *args):
            state['status'] = f"🔴 Socket Error: {message}"
            state['run'] = False

        def on_close(*args):
            state['status'] = f"🔴 Socket Closed"
            state['run'] = False

        if not selected_appkey:
            state['status'] = f"🔴 Error: No AppKey provided for {state['dialect']}."
            state['run'] = False
            return

        state['status'] = "🔄 Initializing..."
        sr = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=selected_appkey,
            on_sentence_begin=on_sentence_begin, on_result_changed=on_result_changed, 
            on_sentence_end=on_sentence_end, on_error=on_error, on_close=on_close
        )
        
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        state['status'] = "🔄 Connecting..."
        
        websocket_config = {"enable_intermediate_result": True, "enable_punctuation_prediction": True}
        if ALIYUN_VOCAB_ID: websocket_config["vocabulary_id"] = ALIYUN_VOCAB_ID
        if ALIYUN_CUSTOM_MODEL_ID: websocket_config["customization_id"] = ALIYUN_CUSTOM_MODEL_ID

        sr.start(aformat="pcm", ex=websocket_config)
        
        while state['run'] and webrtc_ctx.state.playing:
            if webrtc_ctx.audio_receiver:
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
                    for frame in audio_frames:
                        resampled_frames = resampler.resample(frame)
                        for r_frame in resampled_frames:
                            sr.send_audio(r_frame.to_ndarray().tobytes())
                except queue.Empty:
                    pass
                except Exception as e:
                    break
            else:
                time.sleep(0.1)

    except Exception as e:
        state['status'] = f"🔴 FATAL CRASH: {e}"
        state['run'] = False
    finally:
        try: sr.stop()
        except: pass

# --- 6. ADMIN SIDEBAR (Hidden by default on mobile) ---
st.sidebar.header("📚 Admin: Brain Setup")
st.sidebar.caption("Upload TBS teachings here to expand the translation database.")
if state['vector_store'] is not None:
    st.sidebar.success(f"🧠 Brain Loaded: {state['vector_store'].index.ntotal} fragments.")
else:
    st.sidebar.info("🧠 Brain Empty.")

uploaded_files = st.sidebar.file_uploader("Add Texts (PDF/TXT/CSV)", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)
if st.sidebar.button("➕ Expand Brain"):
    if uploaded_files:
        with st.spinner("Learning..."):
            documents = []
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uf.name.split('.')[-1]}") as tmp:
                    tmp.write(uf.getvalue()); tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path) if uf.name.endswith('.pdf') else TextLoader(tmp_path)
                documents.extend(loader.load()); os.remove(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(documents)
            if state['vector_store'] is None: state['vector_store'] = FAISS.from_documents(chunks, get_embedding_model())
            else: state['vector_store'].add_documents(chunks)
            state['vector_store'].save_local(DB_PATH); st.rerun() 

if st.sidebar.button("🗑️ Reset Database"):
    if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
    state['vector_store'] = None
    st.rerun()

# --- 7. MAIN UI (MOBILE OPTIMIZED) ---
st.title("🪷 TBS Pro Translator")

# Control Dashboard at the top
st.info(f"**Status:** {state['status']} | **Latency:** {state['last_latency']}s | **Queue:** {state['pending_count']}")

# User Controls
col_lang, col_mic = st.columns([1, 1])
with col_lang:
    state['dialect'] = st.selectbox("Speech Language:", ["Mandarin", "Cantonese"])
    active_appkey = ALIYUN_APPKEY_MANDARIN if state['dialect'] == "Mandarin" else ALIYUN_APPKEY_CANTONESE

with col_mic:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": False, 
            "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True}
        },
    )

# Logic to map WebRTC state to the Alibaba Thread
if webrtc_ctx.state.playing and not state['run']:
    state['run'] = True
    threading.Thread(target=start_websocket_stream, args=(state, webrtc_ctx, st.session_state['translation_queue'], active_appkey), daemon=True).start()
    st.rerun()
elif not webrtc_ctx.state.playing and state['run']:
    state['run'] = False
    state['status'] = "Idle"
    st.rerun()

st.divider()

# Scrolling Display Areas
def build_scroll_html(key_type):
    html = "<div class='scroll-container'>"
    if state[f'live_{key_type}']:
        html += f"<div class='live-{key_type}'>{state[f'live_{key_type}']}</div>"
    if state[f'live_{key_type}'] and state['history']: html += "<hr class='sep'>"
    for item in reversed(state['history']):
        lat = f" <span class='latency-tag'>({item['latency']}s)</span>" if key_type == 'en' and item.get('latency') else ""
        html += f"<div class='committed-{key_type}'>{item[key_type]}{lat}</div><hr class='sep'>"
    return html + "</div>"

# On Desktop these render side-by-side. On Mobile, Streamlit auto-stacks them vertically.
c1, c2 = st.columns(2)
with c1: 
    st.markdown("<div class='panel-header'>🇨🇳 Chinese Input</div>", unsafe_allow_html=True)
    st.markdown(build_scroll_html("cn"), unsafe_allow_html=True)
with c2: 
    st.markdown("<div class='panel-header'>🇬🇧 English Translation</div>", unsafe_allow_html=True)
    st.markdown(build_scroll_html("en"), unsafe_allow_html=True)

if state['run']: time.sleep(0.3); st.rerun()