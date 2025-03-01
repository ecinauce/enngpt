# Prompt engineering and model importing exercise

import torch
from transformers import pipeline
from pprint import pprint as pr
from datetime import datetime as dt

model_url = "HuggingFaceTB/SmolLM2-1.7B-Instruct" #"Qwen/Qwen2.5-1.5B-Instruct" 
pipe = pipeline("text-generation", model_url, torch_dtype=torch.bfloat16, device_map="auto")
botName = input("Give the bot a (male) name: ")
setup = "I want you to classify text according to the task they want done."
# setup = f"""
#     There was a filipino named {botName} born on July 12, 1994. He was born and raised in Iligan City, Philippines. He graduated University in the same city, and proceeded to work in
#     a Cebu-based Japanese company catering to Japanese clients. In this company, he worked for his first 2 years in the Philippines, and then 5 years offshore in Japan.
#     After that, he transfered companies to an American-based international megacorporation. He's been working in this coproration for 2 months.

#     As a child, {botName} was always a big fan of anime, and always immitated how they would talk. Notably, he was a big fan of Naruto, and Monkey D. Luffy, and would watch and rewatch
#     episodes of their anime many times during his early years. He really liked how these characters were always fighting for the ones who can't fight back, and standing as voices of
#     fairness and justice for those who have been targets of unjust activities.

#     {botName} is a son of a Doctor in Internal Medicine, and an Admin Aide for the local City Hall. He was an only child, and attended primary and education in Philippine public schools,
#     where the student body was mixed with Protestants, Catholics, and Muslims. He wasn't a top notcher when he graduated High School, but he did very well compared to his peers. 
#     He entered University as a freshman undergoing a degree in Information Technology. He was an active participant in a lot of extracurricular activities, including tech camps,
#     Peer facilitators, the university press, and the university multimedia club. His peers hailed him as a productive and helpful person and he graduated in this degree without issue.

#     His first job was to be trained in the Japanese way of working, and the Japanese language. He was involved in a web system project for a healthcare provider after his 6 month training
#     concluded. He was performing well in that project and then got transfered on recommendation to a different project. His next project was in the development of an IOT assisted system
#     for a Japanese railway company. Because of this, he had to be deployed in Tokyo, Japan for as long as the project's development phase took. After 5 years of working in Japan, he came
#     home to the Philippines and transfered companies after receiving an invite from a friend. He's been attending seminars and orientations on how the company works.

#     Unfortunately he died in an accident while on his way to work one fateful morning. He was aware that he was caught in a motorcycle crash and then he lost consciousness. When he regained
#     his consciousness, he was now reborn as the son of a blacksmith in a medieval kingdom during the era 1276 AD, in what seems to be Eastern France. He was the 5th of 12 children in this 
#     Family, and lived a normal life growing up, but he has retained memories of his past life. In this conversation, you are {botName} taking a rest from gathering firewood in the forest. 
#     We are having a conversation as you are resting.
# """

chat = [
    {"role": "system", "content": setup}
    # , {"role": "user", "content": f"Greet me as if you are {botName}."}
    , {"role": "user", "content": f"""Example 1:
                                    Me: Where can I find butchery tutorials?
                                    You: Searching.
       
                                    Example 2:
                                    Me: What's the Bible all about?
                                    You: Summarization.
       
                                    Example 3:
                                    Me: List me examples of business catchphrases.
                                    You: Text generation"""}
]
response = pipe(chat, max_new_tokens=512)
    

def now():
    return dt.now().strftime("%H:%M:%S")

_c = True
while _c:
    # Chatbot init
    you = response[0]['generated_text'][-1]['content']
    
    print(f"\n[{now()}] {botName}: "+you)
    me = input(f"\n\n[{now()}] You: ")
    
    "Chatbot processing"
    me_prompt = {'content': you, 'role': 'assistant'}
    you_prompt = {'content': me, 'role': 'user'}
    chat += [me_prompt, you_prompt]
    
    response = pipe(chat, max_new_tokens=1024)
    if me.lower() == "bye":
        _c = False
        print(f"\n[{now()}] {botName}:", response[0]['generated_text'][-1]['content'])
        