import streamlit as st
import time
import threading
import queue
import re
import os
import tempfile
import json

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

# --- 1. CONFIG & SECRETS ---
ALIYUN_AK_ID = st.secrets["ALIYUN_AK_ID"]
ALIYUN_AK_SECRET = st.secrets["ALIYUN_AK_SECRET"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ALIYUN_APP_MANDARIN = "kNPPGUGiUNqBa7DB"
ALIYUN_APP_CANTONESE = "B63clvkhBpyeehDk"
DB_PATH = "tbs_knowledge_db"

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

st.set_page_config(page_title="TBS Pro Translator", layout="wide")

# --- 2. THE STABLE UI STYLING ---
st.markdown("""
<style>
    .translation-card { background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 20px; min-height: 400px; height: 50vh; overflow-y: auto; }
    .text-display { font-size: 22px; color: #3c4043; line-height: 1.5; margin-bottom: 20px; }
    .target-display { font-size: 22px; color: #1a73e8; line-height: 1.5; font-weight: 500; }
    .live-stream { color: #e67e22; font-style: italic; border-left: 3px solid #e67e22; padding-left: 10px; }
    .latency-tag { font-size: 11px; color: #9aa0a6; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 3. PERSISTENT STATE ---
if 'app_state' not in st.session_state:
    st.session_state['app_state'] = {
        'history': [], 'live_cn': "", 'run': False,
        'vector_store': None, 'dialect': "Mandarin", 'last_latency': 0.0
    }
state = st.session_state['app_state']

@st.cache_resource
def get_brain():
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if os.path.exists(DB_PATH):
            return FAISS.load_local(DB_PATH, model, allow_dangerous_deserialization=True)
    except: return None
    return None

if state['vector_store'] is None:
    state['vector_store'] = get_brain()

# --- 4. TRANSLATION QUEUE & WORKER ---
if 'trans_queue' not in st.session_state:
    st.session_state['trans_queue'] = queue.Queue()
    def translation_worker(q, app_state):
        while True:
            item = q.get()
            if item is None: break
            text, v_store = item
            start_t = time.time()
            try:
                prompt = "You are the official TBS translator. 100% English. Terms: 蓮生活佛=Living Buddha Lian-sheng, 師尊=Grand Master."
                if v_store:
                    docs = v_store.similarity_search(text, k=2)
                    prompt += "\nRef: " + " ".join([d.page_content for d in docs])
                
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                    temperature=0.1
                )
                res = response.choices[0].message.content
                app_state['last_latency'] = round(time.time() - start_t, 2)
                app_state['history'].append({"cn": text, "en": res.strip(), "lat": app_state['last_latency']})
            except: pass
            finally: q.task_done()
    threading.Thread(target=translation_worker, args=(st.session_state['trans_queue'], state), daemon=True).start()

# --- 5. ALIYUN ENGINE ---
def start_aliyun(state, webrtc_ctx, q, appkey):
    try:
        client = AcsClient(ALIYUN_AK_ID, ALIYUN_AK_SECRET, "ap-southeast-1")
        req = CommonRequest(); req.set_method('POST'); req.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        req.set_version('2019-07-17'); req.set_action_name('CreateToken')
        token = json.loads(client.do_action_with_exception(req))['Token']['Id']
        
        def on_sentence_end(message, *args):
            text = json.loads(message)['payload']['result']
            if len(text) >= 2: q.put((text, state['vector_store']))
            state['live_cn'] = ""

        sr = nls.NlsSpeechTranscriber(url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=token, appkey=appkey,
            on_result_changed=lambda m, *a: setattr(state, 'live_cn', json.loads(m)['payload']['result']),
            on_sentence_end=on_sentence_end)
        
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        sr.start(aformat="pcm", ex={"enable_intermediate_result": True, "enable_punctuation_prediction": True})
        
        while state['run'] and webrtc_ctx and webrtc_ctx.state.playing:
            if webrtc_ctx.audio_receiver:
                try:
                    for frame in webrtc_ctx.audio_receiver.get_frames(timeout=0.1):
                        for r_frame in resampler.resample(frame):
                            sr.send_audio(r_frame.to_ndarray().tobytes())
                except: break
            else: time.sleep(0.1)
        sr.stop()
    except: pass

# --- 6. MAIN UI ---
st.title("🪷 TBS Pro Translator")

with st.sidebar:
    st.header("⌨️ Manual Translation")
    m_input = st.text_area("Paste Chinese here:")
    if st.button("Translate Text"):
        if m_input: st.session_state['trans_queue'].put((m_input, state['vector_store']))
    
    st.divider()
    state['dialect'] = st.selectbox("Dialect:", ["Mandarin", "Cantonese"])
    if st.button("Clear History"):
        state['history'] = []
        st.rerun()
    st.divider()
    brain_count = state['vector_store'].index.ntotal if state['vector_store'] else 0
    st.write(f"🧠 Brain Status: {brain_count} items")

# VOICE SECTION
active_key = ALIYUN_APP_MANDARIN if state['dialect'] == "Mandarin" else ALIYUN_APP_CANTONESE
webrtc_ctx = webrtc_streamer(
    key="asr", mode=WebRtcMode.SENDONLY, 
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": False, "audio": True},
    audio_receiver_size=1024 
)

if webrtc_ctx and webrtc_ctx.state.playing and not state['run']:
    state['run'] = True
    threading.Thread(target=start_aliyun, args=(state, webrtc_ctx, st.session_state['trans_queue'], active_key), daemon=True).start()
elif (not webrtc_ctx or not webrtc_ctx.state.playing) and state['run']:
    state['run'] = False

st.divider()

# DISPLAY
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Source (CN)")
    src_h = "<div class='translation-card'>"
    if state['live_cn']: src_h += f"<div class='text-display live-stream'>{state['live_cn']}</div>"
    for i in reversed(state['history']): src_h += f"<div class='text-display'>{i['cn']}</div><hr>"
    st.markdown(src_h + "</div>", 1)

with c2:
    st.markdown("### Translation (EN)")
    tar_h = "<div class='translation-card'>"
    for i in reversed(state['history']):
        st.markdown(f"<div class='target-display'>{i['en']}</div>", 1)
        st.code(i['en'], language="text")
        st.caption(f"Latency: {i['lat']}s")
        st.divider()
    st.markdown("</div>", 1)

if state['run']:
    time.sleep(0.4)
    st.rerun()
