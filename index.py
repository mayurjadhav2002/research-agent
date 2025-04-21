from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_restful import abort, reqparse, Api
from pathlib import Path
import os
from app import understand_research_topic, answer_user_query
import json
from datetime import datetime
from pathlib import Path
import uuid

# Your app and database setup
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.db"
db = SQLAlchemy(app)



api = Api(app)

@app.route("/")
def home():
    return render_template("index.html")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(file, unique_id):
    new_filename = f"{unique_id}_{file.filename}"
    file_path = UPLOAD_FOLDER / new_filename
    file.save(file_path)
    return str(file_path)

@app.route("/generate_research_description", methods=["POST"])
def generate_research_description():
    try:
        description = request.form.get("description")
        unique_id = request.form.get("uuid")
        files = request.files.getlist("files[]")

        uploaded_file_paths = []
        for file in files:
            saved_path = save_uploaded_file(file, unique_id)
            uploaded_file_paths.append(saved_path)

        result = understand_research_topic(description)

        return jsonify({
            "message": "Success",
            "uuid": unique_id,
            "result": result
        })

    except Exception as e:
        return jsonify({"message": "Error"+str(e)}), 500
@app.route("/query_research", methods=["POST"])
def query_research():
    try:
        query = request.form.get('query')

        result = answer_user_query(query)
        if not result:
            abort(404, message="No results found for the given query.")

        return jsonify(result)

    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)
