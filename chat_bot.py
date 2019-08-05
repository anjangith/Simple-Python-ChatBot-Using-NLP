import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
f=open('corpus.txt','r')
raw=f.read()
raw=raw.lower()
sent_tokens=nltk.sent_tokenize(raw)
word_tokens=nltk.word_tokenize(raw)
lemmer=nltk.stem.WordNetLemmatizer()

def lemTokens(tokens):
    return[lemmer.lemattize(tokens) for token in tokens]
	
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def LemTokenize(text):
    return(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETINGS_INPUTS = ("hello","hi","greetings","sup","What's up","hey",)

GREETING_RESPONSES = ["hi","hey","hello","Hi there","I am glad you are taking to me!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETINGS_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[ ]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemTokenize,stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you!"
        return robo_response
    else:
        robo_response=robo_response+sent_tokens[idx]
        return robo_response
        

flag=True
print("My name is ROBO and i will answer your queries about Chatbots. If you want to exit press Bye!")

while(flag==True):
    user_response=raw_input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("ROBO: You are welcome")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO:")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flat=False
        print("ROBO: Bye! Take Care! Ta ta..")


# In[ ]:





# In[ ]:




