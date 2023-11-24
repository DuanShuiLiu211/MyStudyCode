import os
import sys

import markdown
import markdown.extensions.codehilite
import markdown.extensions.fenced_code
from flask import Flask, render_template, request
from markupsafe import Markup

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from Models.model import CallChatGPT3

SESSIONNUM = 1
SESSIONINDEX = 0


demo = Flask(__name__)
gpt_model = CallChatGPT3(
    api_key="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",
    model="gpt-3.5-turbo",
    n=SESSIONNUM,
)


@demo.route("/")
def home():
    return render_template("index.html")


@demo.route("/get_answer", methods=["POST"])
def get_answer():
    user_input = request.form["user_input"]
    answer_list, _ = gpt_model(user_input)

    return Markup(
        markdown.markdown(
            answer_list[SESSIONINDEX], extensions=["fenced_code", "codehilite"]
        )
    )


@demo.route("/reset")
def reset():
    gpt_model.reset_messages()

    return "重新启动对话！"


if __name__ == "__main__":
    demo.run(debug=True)
