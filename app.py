import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

# st.session_state is essentially a dictionary, where keys are used to store values. You can store any type of data in st.session_state, including numbers, strings, lists, dictionaries, and even custom objects

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"bot","text":"Hi,I'm a chatbot who can search the web. How can I help you?"}
        
    # non of these are predefined keywords
    # Each message is a dictionary with keys "role" and "text".
    ]

# add context in session state as key val pair
for msg in st.session_state.messages:
    # st.chat_message(msg["role"]).write(msg['text'])
    if msg["role"] == "user":
        # Display user messages in a bright green
        st.markdown(f"<div style='color: #32CD32; font-weight: bold;'>User: {msg['text']}</div>", unsafe_allow_html=True)
    else:
        # Display bot messages in a bright blue
        st.markdown(f"<div style='color: #1E90FF; font-weight: bold;'>Bot: {msg['text']}</div>", unsafe_allow_html=True)
# : The walrus operator lets you assign a value to a variable and use that value within the same expression. 
# This can be particularly useful in conditions, loops, and list comprehensions where you'd normally need separate lines for assignment and evaluation.
# if (n := len(some_list)) > 10:
#     print(f"The list is too long with {n} elements")
# In this example, the expression n := len(some_list) assigns the length of some_list to the variable n, and then immediately checks whether n is greater than 10.

if prompt:=st.chat_input(placeholder="What is ML?") :
    st.session_state.messages.append({"role":"user","text":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)

    tools=[search,arxiv,wiki]

    # convert tools to agent
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    #  This specifies the type of agent that uses a zero-shot approach, meaning it can react to queries without prior examples.

    with st.chat_message("assistant"):
        # begins a context block where any output generated will be displayed as a message from the assistant (bot).

        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        # you want to update or display the intermediate steps, responses,thoughts , actions of agents

        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'bot',"text":response})
        st.write(response)


