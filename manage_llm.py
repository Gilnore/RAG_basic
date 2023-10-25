# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:34:08 2023

@author: robin

llama 2 is hallucination like some crack-head on cocaine,
maybe GPT-3.5 turbo will be better.

maybe also try fine tuning the local models after getting enough data.

this need a paid account from open ai.

note that huggingface puts a .cache file in user_c
"""
# import memory_data_management as mdm
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM#,TextStreamer
from auto_gptq import AutoGPTQForCausalLM
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
class llm_manager:
    def __init__(self, source = 'open_ai', temp = 0, max_len = 100, local = False, gptq = False):
        load_dotenv()
        openai_key = os.getenv('OPENAI_API_KEY')
        huggingface_key = os.getenv('HUGGINGFACEHUB_API_TOEKN')
        self.source = source
        self.local = local
        if source == 'open_ai' and not  local:
            key = openai_key
            self.llm = ChatOpenAI(
                temperature = temp,
                model = 'gpt-3.5-turbo',
                streaming = True)
            if key is None or key == '':
                input('OPENAI_API_KEY not detected, correct the .env file')
                exit()
        elif local:
            model_dir = os.getenv('pretrained_model_dir')
            model_id = os.getenv('model_base_name')
            source = 'local'
            if model_id is None or model_id == '':
                model_id = source
            if model_dir is None or model_id == '':
                model_dir = input('model directory not found, please enter here:\n')
            if gptq:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True,legacy=False)
                model = AutoGPTQForCausalLM.from_quantized(model_dir,
                                                           model_basename=model_id,
                                                           trust_remote_code=True,
                                                           use_safetensors=True,
                                                           use_triton=False,
                                                           device = 'cuda:0',
                                                           quantize_config=None,
                                                           device_map='auto')
            
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens= max_len,
                    temperature=temp,
                    repetition_penalty=2.
                )
                #generate a langchain llm from pipline
                self.llm = HuggingFacePipeline(pipeline = pipe)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, padding = True)
                model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                             device_map="auto",
                                                             trust_remote_code=False,
                                                             offload_folder = 'offload')
                tokenizer.pad_token_id = model.config.eos_token_id
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens= max_len,
                    temperature=temp,
                    repetition_penalty=2.
                )
                #generate a langchain llm from pipline
                self.llm = HuggingFacePipeline(pipeline = pipe)
        else:
            key = huggingface_key
            model_kwargs = {'temperature':temp,"max_length": max_len}
            self.llm = HuggingFaceHub(repo_id = source,model_kwargs = model_kwargs)

        self.physics_prompt = SystemMessage(content = '''
        you are a physicist who will always think from first principles.
        you will base your opinions on facts, experimental results and logic.
        you will be given information, for which you must determine if they are relavent to the questions asked.\
        you will reason from the information given and ask for more if the information is not sufficent.
        you will always include the referenced data, book chapters, figures and links at the end.
        you will always indicate which part of your response are your own untested conjectures.
        you will designe experiments to test the conjectures or prove them with data.
        you must never make up any data or experimental results.
        you should check your theoretical derivations against existing theory.
        you should check the implications of your theory, and test them against reality.
        ''')
        #the idea is each collection will contain multiple books of the same field
        #one must first find the collection (database) of question
        #then each group in the database will then be the book
        #one will make a database for detailed descriptions of each collection
        '''
        I was planning to use the following as metadata, but this is more suited to semantic searches:
        
        AttributeInfo(name = 'collection summary',
                         description = "a short summary of the book collection's content.",
                         type = 'string')
            
        '''
        self.collection_metadata = [AttributeInfo(name = 'names',
                                                   description = "a list of all names of books the collection contains.",
                                                   type = "list[string]"),
                                     AttributeInfo(name = 'authors',
                                                   description = "a list of all authors of books the collection contains.",
                                                   type = "list[string]"),
                                     AttributeInfo(name = 'years of publishing',
                                                   description = "a list of all years of publishing of books the collection contains.",
                                                   type = "list[int]"),
                                     AttributeInfo(name = 'collection creation data',
                                                   description = 'the date the collection was created',
                                                   type = 'list[int]')]
        #it might be better to implement the above in sql
        #collection summary is an overkill
        #use the rows for the collection
        #each column will contain the above infomation
        #each time a collection is created, make a new row
        #each time a new book is read, add an element to the entries of the row for each column
        #all the texts for a collection is actually placed in a single database
        #with the metadata relavant to each page or chunk
        #try store by page vs store by paragraph
        self.text_metadata = [AttributeInfo(name = 'book name',
                                            description = "name of the book.",
                                            type = "string"),
                              AttributeInfo(name = 'author',
                                            description = 'Author of the book',
                                            type = 'string'),
                              AttributeInfo(name = 'creation data',
                                            description = 'the date the book was written',
                                            type = 'list[int]'),
                              AttributeInfo(name = 'page',
                                            description = "page number.",
                                            type = "int"),
                              AttributeInfo(name = 'images',
                                            description = "dictionary whoes keys are name and description of images on the page,\
                                                and whoes contents are image references on pdfs",
                                            type = "dict{string:string}"),
                              AttributeInfo(name = 'tables',
                                            description = 'list of tables from the page',
                                            type = 'list[dataframe]')
                              ]
            #need to check the actual type of things, dataframes might be called something else
    def chunk(self,text:str, chunk_size = 1000):
        #return a list of documents for some text given
        overlap = np.floor(chunk_size/10)
        text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = chunk_size,
        chunk_overlap  = overlap,
        length_function = len)
        chunks = text_splitter.split_text(text)
        # docs =  [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
        return chunks
    
    def summarize(self, texts, token_len = 1000, min_words = 100, use_orginal_prompt = False,verbose=False):
        '''if this doesn't work, use a different llm'''
        chain_type='map_reduce'
        text_lst = self.chunk(texts,chunk_size=token_len)
        docs = [Document(page_content=t) for t in text_lst[:]]
        if use_orginal_prompt:
            chain = load_summarize_chain(self.llm, 
                                         chain_type=chain_type,
                                         verbose=verbose)
            #this have a tendency of returning nonsensical long essays with incoherent words
        else:
            mapprompt = '''
            you are a analyst who summarizes what is given to you.
            you will base your summary on the texts given and nothing else.
            the summary should be understandable by someone who have never read the original.
            the summary should be less than {max_words} and more than {min_words} in length.
            the summary should be simple.
            the summary must retain the original context.
            ''' .format( max_words = min_words + 100,min_words = min_words) + '''
            
            
            {text} 
            
            ###Response:
            '''
            combineprompt = '''
            you are a analyst who distills what is given to you and form a single summary.
            you will base your summary on the texts given and nothing else.
            the summary should be understandable by someone who have never read the originals.
            the summary should be short.
            the summary should be simple.
            the summary should depend on every element that was given.
            
            {text}
            
            
            ###Response:
            '''
            mapp = PromptTemplate.from_template(template = mapprompt)
            comb = PromptTemplate.from_template(template = combineprompt)
            chain = load_summarize_chain(self.llm, 
                                          chain_type=chain_type, 
                                          map_prompt = mapp,
                                          combine_prompt = comb,
                                          verbose=verbose)
            #this is still not summerizing anything
            #it is not getting any context
            #it is short and consise
        return chain.run(docs)
    def mistral_orca_template(self,text, role = 'System'):
        #System or User
        return f"""<|im_start|> {role}:
            {text}
            <|im_end|>
            """
    def retreive_meta_data(self, text, word_lim = 512):
        if len(text)<word_lim:
            if self.local == True:
                system = self.mistral_orca_template(""" you are writting a reference to the following documment and wishes to obtain some information or meta data from it.
                        you don't want to make anything up, so if you don't know just return "None".
                          """ +  f''' 
                        here is a page where you may find the infomation you're looking for
                        
                        {text}
                        
                        ''')
            #     human = self.mistral_orca_template(""" please create the following list, in this format:
            # ['the title', 'the name or names of the author or authors', 'the subject or genre', 'the date of creation', 'the keywords'],
            # please return only what is asked for and nothing else.""",
            #                                         role = 'User')
            
                human = self.mistral_orca_template(""" please create the following python dictionary, in this format as a string:
            {'title':the title of the book or paper, 'authors':the name or names of the author or authors in a list, 'subject':the subject or genre, 'date':the date of creation or publishing, 'keywords':the keywords in a list}""",
            role = 'User')
                message = [system+human]
                return eval(self.llm.generate(message).generations[0][0].text)
            if self.source == 'open_ai':
                system = SystemMessage(content = """ you are writting a reference to the following documment 
                                       and wishes to obtain some information or meta data from it.
                        you don't want to make anything up, so if you don't know just say you don't know.
                          """ +  f''' 
                        here is a page where you may find the infomation you're looking for
                        
                        {text}
                        
                        ''')
                human = HumanMessage(content = """ please create the following python dictionary, 
                                     in this format as a string:
                                    {'title':the title of the book or paper,
                                     'authors':the name or names of the author or authors in a list,
                                     'subject':the subject or genre, 'date':the date of creation or publishing,
                                     'keywords':the keywords in a list}""")
                message = [[system,human]]
                return eval(self.llm.generate(message).generations[0][0].text)
            else:
                input('no other good enough models yet')
            #     list_of_stuff = ['the title', 'the name or names of the author or authors', 'the subject or genre', 'the date of creation', 'the keywords']
            #     role ="""
            #             you are writting a reference to the following documment and wishes to obtain some information or meta data from it.
            #             you don't want to make anything up, so if you don't know just say you don't know.
            #               """
            #     query = f''' 
            #             here is a page where you may find the infomation you're looking for
                        
            #             {text}
                        
            #             '''
        
            #     message = [role+query+f' what is {wanted} of the document provided?' for wanted in list_of_stuff]
            # return self.llm.generate(message).generations[0][0].text
        else:
            return self.recursive_metadata(text,word_lim=word_lim)

    def generate_itterative_metadata_chain(self, text,dictionary):
        #this custom chain will move through a list of texts
        #and slowly accumulate metadata
        #this method constructs the chain
        system_init_temp = SystemMessagePromptTemplate.from_template('''you are writting a reference to some documment 
                               and wishes to obtain some information or meta data from it.
                you don't want to make anything up, so if you don't know just say you don't know.
                please create it in the following python dictionary format, as a string:
                'title':the title of the book or paper,
                 'authors':the name or names of the author or authors in a list,
                 'subject':the subject or genre, 'date':the date of creation or publishing,
                 'keywords':the keywords in a list''')
        human_init_temp = HumanMessagePromptTemplate.from_template('''
        here is a page where you may find the infomation you're looking for
        {text}
        
        and here is what we already know:
        
        {dictionary}
        
        find what is missing or unknown and fill it in the correct format.
        
        ignoring empty or unknown elements of the dictionary of known stuff,
        
        your response must retain information from what we already know, unless we don't yet know anything.
        
        return only what is needed from the format, do not change the format or their arrangements,
        
        do not add other keys to the dictionary, you can only modify the values.
        ''')
                
        init_prompt = ChatPromptTemplate.from_messages([system_init_temp, human_init_temp])

        #chain is prompt|llm|parser
        init_chain = {'text':itemgetter('text'),
                      'dictionary':itemgetter('dictionary')}|init_prompt|self.llm|StrOutputParser()
        initial_dic = init_chain.invoke({'text':text,'dictionary':dictionary})
        
 #        evaluation_temp = HumanMessagePromptTemplate.from_template('''
 # does the final dictionary {dic_1} contain same or more information as the initial dictionary {dictionary}?
 # does {dictionary} contain the element 'Unknown' or have missing information?
 # if the answer to one of the questions is yes, return True, else return False.
 #                                                                   ''')
 #        system_eval_temp = SystemMessagePromptTemplate.from_template('''
 #                        you are a assitant responsible for evaluating refrences.
 #                        you answer only in True or False.
 #                        return a boolean value of True or False only.
 #                                                                     ''')
 #        eval_prompt = ChatPromptTemplate.from_messages([system_eval_temp,evaluation_temp])
        
 #        eval_chain = {'text':itemgetter('text'),
 #                      'dic_1':init_chain,
 #                      'dictionary':itemgetter('dictionary')}|eval_prompt|self.llm|StrOutputParser()
 #        evaluation = eval_chain.invoke({'text':text,'dictionary':dictionary})
 #        return eval(initial_dic), eval(evaluation)
        init_dict = eval(initial_dic)
        keys = ['title', 'authors', 'subject', 'date', 'keywords']
        keys_are_correct = list(init_dict.keys()) == keys
        if keys_are_correct:
            def missing_ele(ele:str):
                bits = ele.split(' ')
                empty = 'Unknown' in bits or '' in bits or 'unknown' in bits
                return empty
            def missing_in_lst(lst:list):
                if lst != []:
                    return np.sum(np.array([missing_ele(ele) for ele in lst]))
                else:
                    return True
            def dic_quality(dic:dict):
                have_empty = np.array([missing_ele(dic['title']),
                 missing_in_lst(dic['authors']),
                 missing_ele(dic['subject']),
                 missing_ele(dic['date']),
                 missing_in_lst(dic['keywords'])])
                have_empty = np.sum(have_empty) #find out how many unknowns there are
                return have_empty
            old_dic_q = dic_quality(eval(dictionary))
            new_dic_q = dic_quality(init_dict)
            not_regressing = old_dic_q >= new_dic_q
            not_filled = new_dic_q>0
            if not_filled and not not_regressing:
                print('Forgetting error is made by the AI')
                return dictionary, not_regressing and not_filled
            elif not_regressing and not not_filled:
                print('extraction evaluation complete')
                return init_dict, not_regressing and not_filled
            else:
                return init_dict, not_regressing and not_filled
        else:
            print('Formating error is made by the AI')
            return dictionary, False
    def recursive_metadata(self,texts:str,word_lim:int,verbose = False):
        text_chunks = self.chunk(texts,chunk_size=word_lim)
        dic = "{'title':'', 'authors':[''], 'subject':'', 'date':'', 'keywords':['']}"
        ev = True
        for i, txt in enumerate(text_chunks):
            dic,ev = self.generate_itterative_metadata_chain(txt,str(dic))
            if verbose:
                print(dic, ev)
            if not ev and i !=0:
                return dic
        return dic
    def retreive_collection_init(self,query, store, metadata_format,
                                   description = 'these are different sections of a book store',verbose=False):
        ''' this function grabbs as much metadata as it can from the question it recieves,
        it then return them to be used later for retreival.
        use different stores and metadat_formats for different levels.
        might want to change description of what store is fed in as well, such as subbing in book or page
        for book store.
        
        idea is to make a vector store of short descriptions, each with some simpler metadata.
        query that first, getting a list of recommended stores, then search the stores in depth.
        both can be done with this.
        '''
        # metadata_format = self.collection_metadata
        retriever = SelfQueryRetriever.from_llm(self.llm,
                                       store,
                                       description,
                                       metadata_format,
                                       enable_limit=True,
                                       verbose=verbose)
        data = retriever.get_relevant_documents(query)
        #note both content (the summery) and the metadata are returned
        #description or searches on the next level can be constructed from the contents.
        return data
    
    def retreive_conversation_construct(self, store,store_content_description, metadata_format=[],verbose=False):
        ''' this function constructs a conversational retrieval chain.
        the store should contain just one field of pdfs, such as physics, or engineering.
        each store should have an associated description.
        i recomend making a vector store of short descriptions, each with some simpler metadata.
        query that first, getting a list of recommended stores, then search the stores in depth.
        both can be done with this.
        
        store is long term memory
        memory is short term memory
        '''
        retriever = SelfQueryRetriever.from_llm(llm = self.llm,
                                        vectorstore=store,
                                        document_contents = store_content_description,
                                        metadata_field_info = metadata_format,
                                        enable_limit=True,
                                        fix_invalid = True,
                                        verbose=verbose)
        # retriever = store.as_retriever()
        memory = ConversationSummaryBufferMemory(llm=self.llm, 
                                                 max_token_limit=512,
                                                 memory_key="chat_history", 
                                                 return_messages=True, 
                                                 output_key='answer')
        # doc_chain = load_qa_with_sources_chain(llm = self.llm, 
        #                                        chain_type="map_reduce")
        #combine_document_chain = doc_chain cant be used
        #because doc_chain is too complex and the chains conflicts
        conv = ConversationalRetrievalChain.from_llm(llm = self.llm, 
                                                 retriever = retriever, 
                                                 memory = memory,
                                                 return_source_documents = True,
                                                 verbose=verbose)
        #vector store is what you want to retreive infomation from.
        #memory is where the chat history is stored.
        return conv

def main():
    model = llm_manager() 
    #something is different for open_ai
    #Open-Orca/Mistral-7B-OpenOrca not responding
    #models larger than 7b like lmsys/vicuna-33b-v1.3 doesn't respond.
    #anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g didnt respond
    #mistralai/Mistral-7B-Instruct-v0.1 didn't reconize the informations needed
    #try llama 70b when ready: meta-llama/Llama-2-70b-hf
    return model

if __name__ == '__main__':
    mod = main()
    # import PDF_extract as pdfe
    # extractor = pdfe.PDF_extract()
    # pdf = extractor.extract_pdf_test('PDF_store\Generating Efficient Training Data via LLM-based Attribute Manipulation.pdf')
    # text = ''.join(pdf['text'])
    # meta = mod.retreive_meta_data(text)
    # with open('langchain.txt','r') as doc:
    #     text = ''.join(doc.readlines())
    #     print(mod.summarize(text))
    import memory_data_management as mdm
    mem = mdm.mem()
    v_st = mem.load_vector_store()
    conversation = mod.retreive_conversation_construct(v_st,\
                'this is a paper about generating training data for large language models.',
                metadata_format=mod.text_metadata)
    # v_st.similarity_search(query = 'what is this about?')
    print(conversation.invoke('what is this about?'))