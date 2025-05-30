# app.py
import pickle
from flask import Flask, render_template, request
from compute_distances import compute_distances

app = Flask(__name__)

# Load cities from pickle file
with open('cities.pkl', 'rb') as f:
    cities = pickle.load(f)

# Sample graph (keep as needed)
graph_dict = [
    {'start': 'a', 'end': 'b', 'weight': (1, 10, 45)},
    {'start': 'b', 'end': 'c', 'weight': (2, 5, 55)},
    {'start': 'c', 'end': 'd', 'weight': (5, 5, 23)},
    {'start': 'a', 'end': 'd', 'weight': (8, 1, 66)}
]

@app.route('/')
def index():
    return render_template('index.html', cities=cities)

@app.route('/results', methods=['POST'])
def results():
    start = request.form['start']
    end = request.form['end']
    k = int(request.form['k'])

    distances = compute_distances(start, end, graph_dict, k)
    top_k_weights = distances.get(end, [])

    return render_template('results.html', start=start, end=end, k=k, weights=top_k_weights)

if __name__ == '__main__':
    app.run(debug=True)
