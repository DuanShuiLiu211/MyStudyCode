import os
import logging
import openai
from transformers import pipeline


# 加载GPT-2模型
def call_chatgpt2(task="text-generation"):
    generator = pipeline(task=task, model='gpt2')
    
    return  generator


# 加载GPT-3模型
class CallChatGPT3:
    def __init__(self,
                 model="gpt-3.5-turbo",
                 top_p=1,
                 temperature=1,
                 n=1,
                 stream=False,
                 presence_penalty=0,
                 frequency_penalty=0,
                 outputlog_dir=".",
                 outputlog_name="outputs.log"
                 ):
        self.api_key = "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"
        self.model = model
        self.messages = []
        self.temperature = temperature
        self.top_p = top_p
        self.n = n 
        self.stream = stream
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.outputlog_dir = outputlog_dir
        self.outputlog_name = outputlog_name
        self.token_num = 0
    
    def logger(self, content=None):
        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)
        if not log.handlers:      
            os.makedirs(self.outputlog_dir, exist_ok=True)
            filepath = os.path.join(self.outputlog_dir, self.outputlog_name)    
            handler = logging.FileHandler(filename=filepath,
                                          encoding="UTF-8")
            formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            log.addHandler(handler)    
        log.info(content)
        
        return log
    
    def openai_gptapi(self, prompt): 
        openai.api_key = self.api_key
        self.messages.append({"role": "user", "content": prompt})    
        response = openai.ChatCompletion.create(model=self.model,
                                                messages=self.messages,
                                                temperature=self.temperature,
                                                top_p=self.top_p,
                                                n=self.n,
                                                stream=self.stream,
                                                presence_penalty=self.presence_penalty,
                                                frequency_penalty=self.frequency_penalty)
        
        return response
    
    def reset_messages(self):
        self.messages = []
        
    def reset_logger(self):
        filepath = os.path.join(self.outputlog_dir, "outputs.log")
        if os.path.exists(filepath):
            os.remove(filepath)
        self.logger().handlers = []
    
    def __call__(self, prompt):
        self.token_num += 2*(len(prompt)+2)
        if self.token_num > 4000:
            answer = ["即将超过最长对话限制自动重启新的会话"]
            self.reset_messages()
            self.token_num = 0
        else:
            response = self.openai_gptapi(prompt)
            input_string = f"提问: {prompt}"
            self.logger(input_string)
            
            answer = []
            output_content = {i: response.choices[i].message.content for i in range(self.n)}
            for k, v in output_content.items():
                self.token_num += 2*(len(v)+2)
                self.messages.append({"role": "assistant", "content": v})
                output_string = f"回答({k+1}): {v.strip()}\n"
                self.logger(output_string)
                answer.append(v.strip())

        return answer


if __name__ == "__main__":      
    model = CallChatGPT3(temperature=0.8, n=1)
    input_prompt = "你好"
    model(prompt=input_prompt)