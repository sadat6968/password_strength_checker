from flask import render_template, request
from app import app
from app.model import evaluate_password

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        password = request.form["password"]
        strength, suggestions = evaluate_password(password)
        return render_template("result.html", password=password, strength=strength, suggestions=suggestions)
    return render_template("index.html")
