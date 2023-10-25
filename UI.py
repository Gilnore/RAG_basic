# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:45:16 2023

@author: robin
"""
import streamlit as st
from PIL import Image
import io
from langchain.schema import AIMessage, HumanMessage#, SystemMessage
import concurrent.futures as cf

class GUI:
    def __init__(self):
        st.set_page_config(page_title="chat with documents",page_icon=":books:")
        st.header(":books:")
        self.user_q = st.chat_input("ask a question")
        self.state = st.session_state
        self.message_num = 0
        if 'chat_hist' not in self.state:
            self.state['chat_hist'] = []
        st.button('display chat history', on_click=self.display_hist())
        
        # st.button('extract all in dirctory', on_click=extractor.extract_all_in_dirc())
    def status(self,status_prompt, input_prompt=None, warning = False, error = False):
        """call this to display status message or take user inputs"""
        with st.sidebar:
            if warning:
                st.warning(status_prompt)
            elif error:
                st.error(status_prompt)
            else:
                st.toast(status_prompt)
            if not (input_prompt is None):
                ans = st.text_input(input_prompt)
                return ans
    
    def extractor_interface(self, extractor, database_manager):
        #this is currently broken due to streamlit being multithreaded.
        with st.sidebar:
            if st.button('extract entire directory'):
                with st.spinner('extracting directory...'):
                    with cf.ProcessPoolExecutor(max_workers=1) as ppe:
                        processed_job = ppe.submit(extractor.extract_all_in_dirc,database_manager)
                        if processed_job.done():
                            st.success('done')
            if st.button('extract specific content'):
                dirc = st.text_input("what is the directory + name + extension? use directory/name.extension format")
                if not (dirc is None) and dirc != '':
                    with st.spinner('extracting directory...'):
                        texts,im_dicts,tables,metadata = extractor.extract_pdf(dirc)
                        extractor.store_pdf(texts,im_dicts,tables,metadata,database_manager)
                    st.success('done')
    def display_image_bytes(self,loc, base_image,caption):
        #loc is a streamlit container such as a tab or a expander object
        image_bytes = base_image["image"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        loc.image(image = image, caption=caption)
        
    def display_chat(self,mess):
        '''this keeps track of which message is human, updates the message history, and displays the message'''
        self.message_num+=1
        if isinstance(mess,AIMessage) or isinstance(mess,dict):
            message = self.Parse_AI_message(mess)
        if isinstance(mess, HumanMessage):
            with st.chat_message("Human"):
                st.write(mess.content)
            message = mess
        self.state['chat_hist'].append(message)
    def Parse_AI_message(self, message:dict):
        with st.chat_message("AI"):
            st.write(message['answer'])
            sauce = st.expander("source contents")
            for doc in message['source_documents']:
                sauce.write(doc.page_content)
                if 'images' in list(doc.metadata.keys()):
                    images = doc.metadata['images'] #the image dictionary with captions as keys.
                    captions = list(images.keys())
                    for caption in captions:
                        base_image = images[caption]
                        self.display_image_bytes(sauce, base_image, caption)
                if 'tables' in list(doc.metadata.keys()):
                    tables = doc.metadata['tables']
                    for table in tables:
                        sauce.table(table)
            ref = st.expander('references')
            for doc in message['source_documents']:
                ref.write(doc.metadata['title'])
                ref.write(doc.metadata['authors'])
                ref.write(doc.metadata['date'])
                ref.write(doc.metadata['subject'])
            return AIMessage(content = message['answer'])
    def reset_hist(self):
        '''resets the message history'''
        self.message_num = 0
        self.state['chat_hist'] = []

    def display_hist(self):
        '''this will display the entire message history in a tab, with the option to reset.'''
        st.button(label='clear', on_click=self.reset_hist)
        hist = self.state.get('chat_hist',[]) #defaut to [], unless 'chat_hist' is created
        with st.expander('history'):
            for i,ms in enumerate(hist):
                if isinstance(ms,AIMessage):
                    with st.chat_message("AI"):
                        st.write(ms)
                if isinstance(ms, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(ms)