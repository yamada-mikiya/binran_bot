import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.Youtubeing import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

# --- 1. 初期設定とUI ---
st.set_page_config(page_title="神戸大学工学部 学生便覧チャットボット", page_icon="📚")
st.title("📚 神戸大学工学部 学生便覧チャットボット")
st.caption("2024年度の学生便覧をもとに、AIが質問に回答します。")

# --- 2. APIキーの設定 ---
# StreamlitのSecrets機能を利用することを推奨しますが、ここでは直接入力する形式にしています。
try:
    # Streamlit SecretsからAPIキーを読み込む
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # Secretsにない場合はユーザーに入力を促す
    GOOGLE_API_KEY = st.sidebar.text_input("Google AI API Keyを入力してください:", type="password")

if not GOOGLE_API_KEY:
    st.info("サイドバーからGoogle AI APIキーを入力してください。")
    st.stop()

# APIキーを設定
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


# --- 3. データ読み込みとベクトルストアの構築 ---
# @st.cache_resource を使って、重い処理の結果をキャッシュし、アプリの動作を高速化します。
@st.cache_resource
def load_and_build_vector_store():
    """
    学生便覧のテキストを読み込み、チャンクに分割してベクトルストアを構築する関数
    """
    try:
        with open("kobe_u_handbook.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.error("エラー: `kobe_u_handbook.txt` ファイルが見つかりません。`app.py` と同じ階層に配置してください。")
        return None

    # テキストを適切なサイズのチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # チャンクの文字数
        chunk_overlap=200, # チャンク間の重複文字数
        length_function=len,
    )
    text_chunks = text_splitter.split_text(raw_text)

    # Embeddingモデル（テキストをベクトルに変換）の準備
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # ベクトルストア（FAISS）を構築
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# ベクトルストアを読み込み（または構築）
vector_store = load_and_build_vector_store()
if vector_store is None:
    st.stop()


# --- 4. 回答生成AIチェーンの準備 ---
def get_conversational_chain():
    """
    質問応答のためのLangChainチェーンを構築する関数
    """
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
    # 回答生成モデルの準備
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    
    # プロンプトテンプレートの設定
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    
    # チェーンの構築
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- 5. チャットUIの実装 ---
# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "こんにちは！神戸大学工学部の学生便覧について、何でも質問してください。"}]

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの質問入力
if user_question := st.chat_input("質問を入力してください...（例: 早期卒業の条件は？）"):
    # ユーザーの質問を履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("AIが学生便覧を確認しています..."):
            try:
                # 関連ドキュメントをベクトルストアから検索
                docs = vector_store.similarity_search(user_question, k=5)

                # チャット履歴を整形
                chat_history = "\n".join([f"{'Q' if msg['role'] == 'user' else 'A'}: {msg['content']}" for msg in st.session_state.messages[:-1]])

                # QAチェーンを実行して回答を生成
                chain = get_conversational_chain()
                response = chain(
                    {"input_documents": docs, "chat_history": chat_history, "question": user_question},
                    return_only_outputs=True
                )
                answer = response["output_text"]

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info("APIキーが有効か、または利用制限に達していないかご確認ください。")

# --- 6. 注意事項 ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**【注意事項】**\n"
    "このチャットボットは、2024年度の学生便覧の情報を基にAIが回答を生成します。"
    "回答は必ずしも100%正確・完全であることを保証するものではありません。"
    "最終的な確認は、必ず公式の学生便覧で行ってください。"
)