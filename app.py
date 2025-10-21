from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Global DataFrame to hold uploaded data
df = pd.DataFrame(columns=["text", "label"])

def get_ai_label(text):
    """Mock AI labels for testing without using OpenAI"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["great", "awesome", "good", "fantastic"]):
        return "Positive"
    elif any(word in text_lower for word in ["bad", "boring", "terrible", "poor"]):
        return "Negative"
    else:
        return "Neutral"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    global df
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded", 400
        try:
            # Safe reading
            lines = file.read().decode("utf-8").splitlines()
            df = pd.DataFrame(lines, columns=["text"])
            df["label"] = ""  # empty column for labels
        except Exception as e:
            return f"Error reading file: {e}", 500

        df.to_csv("data/uploaded.csv", index=False)
        return render_template("label.html", texts=df["text"].tolist(), labels=df["label"].tolist())

    return render_template("index.html")

@app.route("/label", methods=["POST"])
def save_label():
    global df
    index = int(request.form["index"])
    label = request.form["label"].strip()
    df.at[index, "label"] = label
    df.to_csv("data/labeled_data.csv", index=False)
    return redirect(url_for("upload_file"))

@app.route("/suggest", methods=["GET"])
def ai_suggest():
    global df
    index = int(request.args.get("index"))
    text = df.at[index, "text"]
    ai_label = get_ai_label(text)
    return render_template("label.html",
                           texts=df["text"].tolist(),
                           labels=df["label"].tolist(),
                           suggest_index=index,
                           ai_label=ai_label)

if __name__ == "__main__":
    app.run(debug=True)
