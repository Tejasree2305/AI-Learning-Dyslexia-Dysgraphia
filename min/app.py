import google.generativeai as genai
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------
# TECHNICAL (NOT RECOMMENDED, BUT WORKS)
# ---------------------------------------
def init_gemini():
    api_key = "AIzaSyDCUSw8Cnf780l8r-lkYaAlznhY-_fruDg"
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

model = init_gemini()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        prompt = """
Generate a 4-week learning plan.
For each week include:
Reading Activity
Writing Activity
Motor Skills Activity
Assessment Method
"""

        response = model.generate_content(prompt)
        result = response.text.split("\n")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
