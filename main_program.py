# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:35:44 2023

@author: robin
"""
import UI
import memory_data_management as mdm
import PDF_extract as pdfe
import manage_llm as mllm
import streamlit as st
from langchain.schema import AIMessage, HumanMessage, SystemMessage
@st.cache_resource()
def initializing_program(_ui):
    print('initiating memory manager')
    memo = mdm.mem(UI = ui)
    ui.state['mem_manage'] = memo
    print('memory manager stored')
    ui.state['existing client'] = memo.client
    del memo
    print('deleting original')
    print('loading vector store')
    vector_store = ui.state['mem_manage'].load_vector_store()
    print('storing vector store')
    ui.state['vect-store'] = vector_store
    print('loading pdf extractor')
    extractor = pdfe.PDF_extract(UI = ui)
    print('saving extractor')
    ui.state['pdf extractor'] = extractor
    del extractor
    print('loading llm manager')
    mod = mllm.llm_manager()
    print('storing memory manager')
    ui.state['llm manager'] = mod
    print('deleting llm manager')
    del mod
    print('initiating conversation')
    conv = ui.state['llm manager'].retreive_conversation_construct(ui.state['vect-store'],'this is everything you know', metadata_format = ui.state['llm manager'].text_metadata)
    return conv
ui = UI.GUI()
conv = initializing_program(ui)
# ui.extractor_interface(ui.state['pdf extractor'],ui.state['mem_manage'])
print('detecting query')
if ui.user_q:
    print('query detected')
    query = HumanMessage(content = ui.user_q)
    print('displaying')
    ui.display_chat(query)
    print('generating response')
    with st.spinner("generating reponses:"):
        response = conv.invoke(ui.user_q)
    print('displaying')
    ui.display_chat(response)
