import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os


# --- 1. 初期設定とUI ---
st.set_page_config(page_title="神戸大学工学部 学生便覧チャットボット", page_icon="📚")
st.title("📚 神戸大学工学部 学生便覧チャットボット")
st.caption("2024年度の学生便覧をもとに、AIが質問に回答します。")

# --- 2. APIキーの設定 ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    GOOGLE_API_KEY = st.sidebar.text_input("Google AI API Keyを入力してください:", type="password")

if not GOOGLE_API_KEY:
    st.info("サイドバーからGoogle AI APIキーを入力してください。")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Google APIキーの設定中にエラーが発生しました: {e}")
    st.stop()

# --- 3. データ読み込みとベクトルストアの構築 ---
@st.cache_resource
def load_and_build_vector_store():
    try:
        with open("kobe_u_handbook.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.error("エラー: `kobe_u_handbook.txt` ファイルが見つかりません。")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(raw_text)
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # ← ここも修正
        return vector_store
    except Exception as e:
        st.error(f"ベクトルストアの構築中にエラーが発生しました。APIキーやライブラリの互換性を確認してください。エラー詳細: {e}")
        return None


vector_store = load_and_build_vector_store()
if vector_store is None:
    st.stop()

# --- 4. 回答生成AIチェーンの準備 ---
def get_conversational_chain():
    prompt_template = """
    あなたは神戸大学工学部の学生便覧に関する質問に答える、親切で優秀なアシスタントです。
    以下の「コンテキスト」情報と「チャット履歴」を元に、「質問」に対して日本語で詳しく、丁寧に回答を生成してください。
    回答は、学生にとって分かりやすいように、必要であれば箇条書きなどを用いて整理してください。
    コンテキストから答えが見つからない場合は、無理に答えを作成せず、「学生便覧の情報からは回答が見つかりませんでした。」と明確に伝えてください。
    必ずコンテキストの内容に基づいて回答し、一般的な知識で答えないでください。

    【コンテキスト】
    {context}

    【チャット履歴】
    {chat_history}

    【質問】
    {question}

    【回答】
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- 5. チャットUIの実装 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "こんにちは！神戸大学工学部の学生便覧について、何でも質問してください。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("質問を入力してください...（例: 早期卒業の条件は？）"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("AIが学生便覧を確認しています..."):
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(user_question)

                chat_history = "\n".join([f"{'Q' if msg['role'] == 'user' else 'A'}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                chain = get_conversational_chain()
                response = chain(
                    {"input_documents": docs, "chat_history": chat_history, "question": user_question},
                    return_only_outputs=True
                )
                answer = response["output_text"]

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"回答生成中にエラーが発生しました: {e}")

# --- 6. 注意事項 ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**【注意事項】**\n"
    "このチャットボットは、2024年度の学生便覧の情報を基にAIが回答を生成します。"
    "回答は必ずしも100%正確・完全であることを保証するものではありません。"
    "最終的な確認は、必ず公式の学生便覧で行ってください。"
)
