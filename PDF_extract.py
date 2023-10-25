# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:29:03 2023

@author: robin
"""
import fitz as ft
import tabula as ta
import numpy as np
import string
import manage_llm as mllm
import os
from dotenv import load_dotenv
import glob
import memory_data_management as mdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class PDF_extract:
    #this class can currently extract a pdf of mostly texts
    #we need to find a AI that can generate text from image
    #the text will need to accuratly describe what is happening in the image
    #any text or maths in the image need to be converted to text
    #we might want to make our own AI for that.
    def __init__(self, UI = None, local = True):
        load_dotenv() #load in the environment
        # self.path = os.getenv("sql_local_path")
        # nopath = self.path is None or self.path == ''
        dircs = os.getenv("pdf_store_dir")
        nodirc = dircs is None or dircs == ''
        self.UI = UI
        self.model = mllm.llm_manager() # use local = local if a good model pops up for metadata extraction
        if nodirc:
            if UI == None:
                dircs = input("pdf folder not specified, type in the directory here: \n")
            else:
                dircs = UI.status("pdf folder not specified",input_prompt = " type in the directory here: \n")
        else:
            self.dir_lst = glob.glob(dircs+'\\*.pdf')
        # if nopath:
        #     if UI == None:
        #         self.path = input("sql folder not specified, type in the directory here: \n")
        #     else:
        #         self.path = UI.status("sql folder not specified", input_prompt = " type in the directory here: \n")

    # def remove_image_text(self,txt):
    #     #this function currently can't find anything
    #     #it's possible what we're looking for isn't in text
    #     s = txt.find('<image:')
    #     e = txt.find('>')
    #     if s==-1:
    #         return txt[:s]+txt[e:]
    #     else:
    #         print('dangerously removing image reference from text')
    #         self.remove_image_text(txt)
    def process_text(self,text_block, special_character_fractional_tolorance = 0.4):
        #clean up the text a bit
        if text_block[-1] == 0:
            if "�" in text_block[4]:
                if self.UI == None:
                    print('possible missing fonts')
                else:
                    self.UI.status('possible missing fonts', warning = True)
                text = ''.join([char for char in text_block[4] if char != "�"]) #remove unknown characters
                text = text.strip().replace('-\n','').replace('\n',' ').replace('NaN', ' ')
            else:
                text = text_block[4].strip().replace('NaN', ' ').replace('-\n','').replace('\n',' ')
            special = ''.join([char for char in text if char=="\n" or char==" " or char in string.punctuation])
            not_mostly_special_char = max(len(special),1)/max(len(text),1) <special_character_fractional_tolorance
            if not not_mostly_special_char:
                if self.UI == None:
                    print('text is not legible as english')
                else:
                    self.UI.status('text is not legible as english', warning = True)
            txt_top_left = np.array([text_block[0],text_block[1]])
            return not_mostly_special_char, text+'\n\n', txt_top_left
        else:
            return False, '',np.array([np.nan,np.nan])
    
    def process_page_txt(self, page, special_character_fractional_tolorance = 0.4):
        #clean up the entire page
        #return the information that i need as a zipped object of parallel lists
        blocks = page.get_text("blocks")
        valid = []
        text_content = []
        text_box_top_left_corners = []
        for block in blocks:
            not_mostly_special_char, text, txt_top_left = self.process_text(block,special_character_fractional_tolorance)
            valid.append(not_mostly_special_char)
            text_content.append(text)
            text_box_top_left_corners.append(txt_top_left)
        return [valid, text_content, text_box_top_left_corners]
    
    def extact_image_context(self,page,im_ref, page_info, horisontal_tolorance = 90):
        #page is a pymupdf Page object
        #ref is a full xref tuple from get_image()
        #extract bottom left corner of image
        image_loc = page.get_image_bbox(im_ref) #this need full = True on get_image()
        image_bottom_left = np.array([image_loc[0],image_loc[3]])
        image_top_left = np.array([image_loc[0],image_loc[1]])
        #extract upper left corner of text
        key = lambda info: info[1][1]
        page_info_s = page_info.copy()
        page_info_s.sort(key=key)
        text_locs = [(info[0],info[1]) for info in page_info_s]
        #search for nearest in distance to bbox of image
        closest = 'un-named {}'.format(str(im_ref[7]))
        closest_r = np.inf
        for block in text_locs:
            txt_top_left = block[1]
            text_cont = block[0]
            if image_top_left[1]>=txt_top_left[1] or abs(image_top_left[0]-txt_top_left[0]) > horisontal_tolorance:
                continue
            r = abs(sum(image_bottom_left**2)**.5 - sum(txt_top_left**2)**.5)
            if r < closest_r:
                # print(r)
                if 'fig' in text_cont.lower() or 'curve'in text_cont.lower()\
                    or 'photo'in text_cont.lower() or 'picture'in text_cont.lower():
                    closest = text_cont
                closest_r = r
            else:
                break
        return closest
        #return the text
    # def extract_pdf_test(self, dirc, special_character_fractional_tolorance = 0.4, use_AI = False):
    #     texts = []
    #     tables = [] #list of lists, a list of tables per page
    #     im_dict = {}
    #     with ft.open(dirc) as pdf:
    #         for i, page in enumerate(pdf):
    #             print("reading page {}".format(i))
    #             page_text_info = self.process_page_txt(page,special_character_fractional_tolorance) #clean up the pages
    #             image_ref = page.get_images(full=True) #grab all image references of the page
    #             if True in page_text_info[0]:
    #                 valid_info = [(page_text_info[1][i],page_text_info[2][i]) 
    #                               for i, info in enumerate(page_text_info[0])
    #                               if info]
    #                 if image_ref != []:
    #                     for ref in image_ref:
    #                         context = self.extact_image_context(page, ref, valid_info)
    #                         im_dict[context] = ref
    #                 page_txt = "".join([text[0] for text in valid_info])
    #                 texts.append(page_txt)
    #                 table = ta.read_pdf(dirc, multiple_tables=True, pages = i+1)
    #                 #extract tables of data
    #                 if table != []:
    #                     tables+=table
    #             elif use_AI:
    #                 print("can't extract context algorithmically, as page text can't be reconized")
    #                 print("attempt to create context with AI vision")
    #             else:
    #                 print("can't extract context algorithmically, as page text can't be reconized")
    #         metadata = pdf.metadata
    #     return {'text':texts, 'table':tables, 'images':im_dict, 'meta_data':metadata}
    def embded_instruct_template(self,title:str,subjects:list[str],keywords:list[str])->str:
        return f"""
        represent the following document from article {title} of the subjects: {subjects} related to the following {keywords}.
        """
    def extract_pdf(self, dirc, special_character_fractional_tolorance = 0.4):
        with ft.open(dirc) as pdf:
            raw_metadata = pdf.metadata
            texts = []
            tables = {}
            im_dicts = {}
            for i, page in enumerate(pdf):
                print("reading page {}".format(i))
                page_text_info = self.process_page_txt(page,special_character_fractional_tolorance) #clean up the pages
                image_ref = page.get_images(full=True) #grab all image references of the page
                if True in page_text_info[0]:
                    #if there is text on the page
                    valid_info = [(page_text_info[1][j],page_text_info[2][j]) 
                                  for j, info in enumerate(page_text_info[0])
                                  if info]
                    if image_ref != []:
                        im_dict = {}
                        for ref in image_ref:
                            context = self.extact_image_context(page, ref, valid_info)
                            image = pdf.extract_image(ref[0])
                            im_dict[context] = image
                        im_dicts[i] = im_dict
                    page_txt = "".join([text[0] for text in valid_info])
                    texts.append(page_txt)
                    table = ta.read_pdf(dirc, multiple_tables=True, pages = i+1)
                    #extract tables of data
                    if table != []:
                        tables[i]=table #joining the lists
            # book_summary = self.model.summarize(''.join(pages_summeries))
            #most metadata are contained in the first 5 pages
            if len(texts) > 6:
                meta_extract_txt = ''.join(texts[:6])
            else:
                meta_extract_txt = texts[0]
            metadata =self.model.retreive_meta_data(meta_extract_txt)
            #inject metadata
            if isinstance(metadata, list):
                if raw_metadata['title'] == '' or raw_metadata['title'] is None:
                    raw_metadata['title'] = metadata[0]
                if raw_metadata['author'] == '' or raw_metadata['author'] is None:
                    raw_metadata['author'] = metadata[1]
                if raw_metadata['subject'] == '' or raw_metadata['subject'] is None:
                    raw_metadata['subject'] = metadata[2]
                if raw_metadata['creationDate'] == '' or raw_metadata['creationDate'] is None:
                    raw_metadata['creationDate'] = metadata[3]
                if raw_metadata['keywords'] == '' or raw_metadata['keywords'] is None:
                    raw_metadata['keywords'] = metadata[4]
                metadata = raw_metadata
            #now that everything is ready, time to store the texts with the metadata
            #instructions for the embeddings will depend on title subject and keywords of the book
            return {'text':texts,'images':im_dicts,'table':tables,'metadata':metadata}
    def store_pdf(self,texts,im_dicts,tables,metadata, vector_data_base:mdm.mem, summarize = False):
        # self.vdb = vector_data_base # mdm.mem()
        for i, txt in enumerate(texts):
            title = metadata['title']
            subjects = str(metadata['subject'])
            keywords = str(metadata['keywords'])
            embed_instruction = self.embded_instruct_template(title, subjects, keywords)
            if i in im_dicts.keys():
                metadata['images'] = im_dicts[i]
            if i in tables.keys():
                metadata['tables'] = tables[i]
            # self.vdb.perm_mem(txt,meta=metadata,embed_instruction=embed_instruction)
            vector_data_base.perm_mem(txt,meta=metadata,embed_instruction=embed_instruction)
            if summarize:
                summary = self.model.summarize(txt)
                print(summary)
        # return {'text':summaries, 'table':tables, 'images':im_dict, 'meta_data':metadata}
        return {'text':texts,'images':im_dicts,'table':tables,'metadata':metadata}
    def extract_all_in_dirc(self, database, multiprocess = True):
        # txt_dict = {}
        #each directory is an collection
        #create a dictionary {name:{'contents':contents}}
        #we'll consider embedding the name with some summary of the book jn a vector database
        #embed the actual stuff in vector databases with these names
        #then any search will be for the name and references first
        #then the querys will be for the txts within the sub vector databases
        extract = self.extract_pdf
        # print(self.dir_lst)
        if multiprocess:
            #this uses process pool executor to
            # name = lambda dirc: dirc.split("\\")[-1].split(".")[0]
            # names = [dirc.split("\\")[-1].split(".")[0] for dirc in self.dir_lst]
            with ProcessPoolExecutor() as ppe:
                submits = [ppe.submit(extract, dirc) for dirc in self.dir_lst]
                results = [submit.result() for submit in as_completed(submits)]
                text_dicts = results
            #     txts = list(ppe.map(extract, self.dir_lst))
            # txt_dict = {names[i]:txts[i] for i in range(len(names))}
        else:
            text_dicts = [extract(dirc) for dirc in self.dir_lst]
        for dicts in text_dicts:
            self.store_pdf(dicts['text'], dicts['images'], dicts['table'], dicts['metadata'],database)
        print('finished')
        return text_dicts
    
    # def store_additional_contents(self, nam,im_dict,tables,conn):
    #     #the following creates the table
    #     pd.DataFrame.from_dict(im_dict).to_sql(name = nam + "_figures", con = conn, if_exists = 'replace')
    #     #when ever the words related to figures appears, query keys
    #     for i, table in enumerate(tables):
    #         for j, tb in enumerate(table):
    #             tb.to_sql(name = nam + f"_table{(i,j)}", con = conn, if_exists = 'replace') #when words tables or lists appears, query the entire thing
    #     print("stored")
        
    # def extract_images(self,dirc, image_refs:list)->list:
    #     with ft.open(dirc) as pdf:
    #         image_set = [pdf.extract_image(ref[0]) for ref in image_refs]
    #     return image_set
    # #extracting the image dosen't tell you what it is. use create instead
    # def make_images(self, dirc, image_refs:list)->list:
    #     #this is for a single image ref object (...,....,...)
    #     with ft.open(dirc) as pdf:
    #         return [ft.Pixmap(pdf,ref[0]) for ref in image_refs]
    #we'll only store image references, then extract from pdf when retreiving
def main():
    import memory_data_management as mdm
    database = mdm.mem()
    pdfs = PDF_extract()
    return pdfs.extract_all_in_dirc(database)
    # return pdfs.extract_pdf('PDF_store\Generating Efficient Training Data via LLM-based Attribute Manipulation.pdf')

if __name__ == "__main__":
    pdf = main()