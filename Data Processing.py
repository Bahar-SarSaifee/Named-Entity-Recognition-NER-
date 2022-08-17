#!/usr/bin/env python
# coding: utf-8

# # NER using Spacy

# In[1]:


import spacy
from spacy import displacy

NER = spacy.load("en_core_web_sm")


# In[2]:


text = "Named entity recognition (NER) is the task to identify mentions of rigid designators from text belonging to predefined semantic types such as person, location, organization etc. NER always serves as the foundation for many natural language applications such as question answering, text summarization, and machine translation. Early NER systems got a huge success in achieving good performance with the cost of human engineering in designing domain-specific features and rules. In recent years, deep learning, empowered by continuous real-valued vector representations and semantic composition through nonlinear processing, has been employed in NER systems, yielding stat-of-the-art performance. In this paper, we provide a comprehensive review on existing deep learning techniques for NER. We first introduce NER resources, including tagged NER corpora and off-the-shelf NER tools. Then, we systematically categorize existing works based on a taxonomy along three axes: distributed representations for input, context encoder, and tag decoder. Next, we survey the most representative methods for recent applied techniques of deep learning in new NER problem settings and applications. Finally, we present readers with the challenges faced by NER systems and outline future directions in this area."


# In[3]:


NER_text= NER(text)


# In[4]:


for word in NER_text.ents:
    print(word.text,word.label_)
    print(spacy.explain(word.label_), '\n')


# ### Shows the NEs directly in the text

# In[6]:


displacy.render(NER_text,style="ent",jupyter=True)


# ### NER of a News Article

# In[7]:


# importing required modules 
import PyPDF2 
    
# creating a pdf file object 
pdfFileObj = open('paper.pdf', 'rb') 
    
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    
# printing number of pages in pdf file 
print(pdfReader.numPages) 
    
# creating a page object 
pageObj = pdfReader.getPage(0) 
    
# # extracting text from page 
# print(pageObj.extractText()) 
    
text = pageObj.extractText()
    
# closing the pdf file object 
pdfFileObj.close() 


# In[8]:


NER_text = NER(text)


# In[9]:


displacy.render(NER_text ,style="ent",jupyter=True)


# In[ ]:





# In[ ]:




