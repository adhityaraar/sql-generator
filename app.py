import os
import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from PIL import Image
from googletrans import Translator
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
# from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.prompts import ChatPromptTemplate
from langchain.utilities import SQLDatabase

load_dotenv()

api_key = os.getenv("WX_API_KEY", None)
ibm_cloud_url = os.getenv("WX_PROJECT_ID", None)
project_id = os.getenv("WX_URL", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    credentials = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

st.set_page_config(
    page_title="DBS Rucika Indonesia",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

header_text = 'DBS Rucika Indonesia <span style="color: blue; font-family: Cormorant Garamond; font-size: 40px;">| Watsonx</span>'
st.markdown(f'<h1 style="color: black;">{header_text}</h1>', unsafe_allow_html=True)

with st.sidebar:
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='watsonx.ai, a next generation enterprise studio for AI builders to train, validate, tune and deploy AI models')

    st.write("Configure model and parameters:")

    model_option = st.selectbox("Model Selected:", ["granite-13b", "flan-ul2"])
    max_new_tokens = st.number_input("Max Tokens:", 1, 256, value=128)
    min_new_tokens = st.number_input("Min Tokens:", 0, value=8)
        
    st.markdown('''
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) [LLM model](https://python.langchain.com/docs/get_started/introduction)
    ''')

if model_option == "flan-ul2":
    model_selected = ModelTypes.FLAN_UL2.value
else:
    model_selected = ModelTypes.GRANITE_13B_CHAT.value

st.markdown('<hr style="border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

def translate_to_bahasa(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Bahasa based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Bahasa. If False, translates to English.
    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Bahasa
            translate = translator.translate(sentence, dest='id')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e

### Get the database
db = SQLDatabase.from_uri("mysql+pymysql://root:<password>@<localhost>:<port>/<schema>")

def get_schema(_):
    return db.get_table_info(["RTL","SJR"])

def run_query(query):
    return db.run(query)

### Get the model
parameters_starcode = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: 80,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

parameters_flan = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MAX_NEW_TOKENS: max_new_tokens,
    GenParams.MIN_NEW_TOKENS: min_new_tokens,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

starcoder_model = Model(
    model_id=ModelTypes.STARCODER, 
    params=parameters_starcode, 
    credentials=credentials,
    project_id=project_id)

model_lang_selected = Model(
    model_id=model_selected, 
    params=parameters_flan, 
    credentials=credentials,
    project_id=project_id)

starcoder_llm = WatsonxLLM(model=starcoder_model)
model_lang = WatsonxLLM(model=model_lang_selected)

template_code = """Based on the table schema below, write a postgres query that would answer the user's question. Follow SQL programming syntax.
Schema: {schema}

Question: What is the total purchase amount of item B and item C in the RTL channel?
SQL Query: SELECT SUM(itemQty * itemPrice) AS TotalPurchaseAmount FROM rucika.RTL WHERE itemProduct IN ('Item B', ' Item C');

Question: What is total price of item A and item G?
SQL Query: SELECT SUM(itemQty * itemPrice) AS TotalPrice FROM (SELECT itemQty, itemPrice FROM rucika.RTL WHERE itemProduct IN ('Item A', 'Item G') UNION ALL SELECT itemQty, itemPrice FROM rucika.SJR WHERE itemProduct IN ('Item A', 'Item G')) AS combinedTables;

Question: How many customer ordered in February 2023?
SQL Query: SELECT COUNT(DISTINCT customer) AS NumberOfCustomers FROM (SELECT customer FROM rucika.RTL WHERE orderDate >= '2021-02-01' AND orderDate <= '2021-02-28' UNION SELECT customer FROM rucika.SJR WHERE orderDate >= '2021-02-01' AND orderDate <= '2021-02-28') AS combinedOrders;

Question: {question}
SQL Query:
"""

template_lang = """
When presented with a table schema, a specific question, an SQL query, and the corresponding SQL response, your task is to compose a natural language response that comprehensively addresses all aspects of the question using the information provided in the SQL responses. For questions that inquire about price, ensure to present the amount in the Indonesian Rupiah (IDR) format. In cases where the SQL query results in an error or returns no data, respond with 'I don't know' and request further clarification to resolve the issue.

Schema:{schema}
Question: {question}
SQL Query: {query}
SQL Response: {response}
Natural Language Response:
"""

prompt = ChatPromptTemplate.from_template(template_code)
prompt_response = ChatPromptTemplate.from_template(template_lang)

def list_query_parser(queries):
    query_list= queries.splitlines()
    return [i for i in query_list if i]

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Halo, Silahkan tanya apa saja terkait penjualan order Anda!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("Send a message...", key="prompt"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    print(f"\n{user_question}\n")
    translated_user_input = translate_to_bahasa(user_question, False)
    print(f"{translated_user_input}\n")


if st.session_state.messages[-1]["role"] != "assistant":

    with st.chat_message("assistant"):
        with st.spinner("Harap Tunggu..."):

            sql_response = (
                RunnablePassthrough.assign(schema=get_schema)
                | prompt
                | starcoder_llm.bind(stop=["Question"])
                | StrOutputParser()
                | list_query_parser
            )

            try:
                full_chain = (
                    RunnablePassthrough.assign(query=sql_response) 
                    | RunnablePassthrough.assign(
                        schema=get_schema,
                        response = lambda x: [db.run(i) for i in x["query"]]
                    )
                    | prompt_response 
                    | model_lang
                )

                translated_user_input = translate_to_bahasa(user_question, False)

                sql_generator = sql_response.invoke({"question": translated_user_input})
                print(f"{sql_generator}\n")

                response = full_chain.invoke({"question": translated_user_input})
                print(f"{response}\n")

                if "<|endoftext|>" in response:
                    response = response.replace("<|endoftext|>", "")

                response = translate_to_bahasa(response, True)
                print(f"{response}\n")

            except:
                response = "Maaf, saya tidak mengerti pertanyaan Anda. Silahkan tanya kembali dengan pertanyaan yang lebih spesifik."
                print("Error: Cannot generate SQL query")

        placeholder = st.empty()
        full_response = ''

        for item in response:
            full_response += item
            placeholder.markdown(full_response)
        placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
