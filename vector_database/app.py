from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)
DATABASE_FILE = "vector_database.pkl"


def carregar_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "rb") as f:
            database = pickle.load(f)
            #print("Base de dados carregada:", database) 
            return database
    return []

def salvar_database(database):
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)


@app.route("/vectors", methods=["GET"])
def get_vectors():
    database = carregar_database()
    return jsonify(database)

@app.route("/vector", methods=["POST"])
@app.route("/vector", methods=["POST"])
def add_vector():
    data = request.get_json()
    if "name" not in data or "vector" not in data:
        return jsonify({"error": "Invalid data"}), 400
    
    if not isinstance(data["vector"], list) or not all(isinstance(i, (int, float)) for i in data["vector"]):
        return jsonify({"error": "Vector must be a list of numbers"}), 400
    
    database = carregar_database()
    database.append((data["name"], data["vector"]))
    salvar_database(database)
    return jsonify({"message": "Vector added successfully"}), 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
