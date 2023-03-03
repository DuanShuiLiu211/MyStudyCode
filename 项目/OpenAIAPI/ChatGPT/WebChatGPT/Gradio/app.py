import os
import sys
import markdown
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from Models.model import CallChatGPT3

model = CallChatGPT3(temperature=0.8, n=1)


def chatbot_interaction_1st(question, messages):
    messages = messages or []
    answer = model(question)[0]  
    messages.append((question, answer))
    text = ""
    
    return text, messages, messages


def reset_session_1st():
    model.reset_messages()
    messages = None
    text = ""
    
    return text, text, messages


def chatbot_interaction_2st(question):
    answer = model(question)[0]
    # TODO: Latex渲染问题的缓解方案
    answer = answer.replace("$\sqrt{}$", "√")
    answer = answer.replace("$\sqrt{ }$", "√")
    text = ""
        
    return text, answer


def reset_session_2st():
    model.reset_messages()
    answer = ""
    text = ""
    
    return text, answer


def load_text(filename):
    if not filename:
        filename = "outputs.log"
    with open(f"{filename}", "r") as f:
        messages = f.read()
    # TODO: Latex渲染问题的缓解方案
    messages = messages.replace("$\sqrt{}$", "√")
    messages = messages.replace("$\sqrt{ }$", "√") 
    text = ""
    
    return text, messages


def clear_text(text):
    text = ""
    
    return text, text


def detele_text(text):
    model.reset_logger()
    text = ""
    
    return text, text



with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Chatbot</h1>Welcome To Play - Support By OpenAI</center>")
    with gr.Group():
        with gr.Box():
            with gr.Tab("交互"):
                in_text_1st = gr.Textbox(label="输入",
                                         show_label=False,
                                         lines=5,
                                         max_lines=10,
                                         placeholder="输入你的问题")
                out_text_1st = gr.Chatbot()

                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            start_btn_1st = gr.Button("开始")  
                        with gr.Column():
                            reset_btn_1st = gr.Button("重启")                   
            
            with gr.Tab("经典"):
                with gr.Row():
                    with gr.Box():
                        with gr.Column():
                            in_text_2st = gr.Textbox(label="输入",
                                                    show_label=False,
                                                    lines=20,
                                                    max_lines=20,
                                                    placeholder="输入你的问题")
                    with gr.Box():
                        with gr.Column():
                            out_text_2st = gr.Markdown()
                
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            start_btn_2st = gr.Button("开始")
                        with gr.Column():
                            reset_btn_2st = gr.Button("重启") 

                                         
            with gr.Tab("日志"):      
                log_text_1st = gr.Textbox(label="输入",
                                            show_label=False,
                                            lines=1,
                                            max_lines=2,
                                            placeholder="输入日志名称")
                with gr.Box():
                    log_text_2st = gr.Markdown()
                    
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            load_btn = gr.Button("导入")
                        with gr.Column():
                            clear_btn = gr.Button("清除")
                        with gr.Column():
                            detele_btn = gr.Button("删除")
        
        # 一、交互
        init_state = gr.State([])
        start_btn_1st.click(chatbot_interaction_1st, inputs=[in_text_1st, init_state], outputs=[in_text_1st, out_text_1st, init_state])
        reset_btn_1st.click(reset_session_1st, inputs=[], outputs=[in_text_1st, out_text_1st, init_state])
        
        # 二、经典
        start_btn_2st.click(chatbot_interaction_2st, inputs=[in_text_2st], outputs=[in_text_2st, out_text_2st])
        reset_btn_2st.click(reset_session_2st, inputs=[], outputs=[in_text_2st, out_text_2st])
        
        # 三、日志
        load_btn.click(load_text, inputs=[log_text_1st], outputs=[log_text_1st, log_text_2st])
        clear_btn.click(clear_text, inputs=[log_text_1st], outputs=[log_text_1st, log_text_2st])
        detele_btn.click(detele_text, inputs=[log_text_1st], outputs=[log_text_1st, log_text_2st])

    
demo.launch(debug=True)
