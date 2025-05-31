from flask import Flask, render_template, request
from compute_distances import compute_distances
import pickle

app = Flask(__name__)

# Load cities and graph
with open('cities.pkl', 'rb') as f:
    cities = pickle.load(f)

with open('graph.pkl', 'rb') as fp:
    graph_dict = pickle.load(fp)

@app.route('/')
def index():
    return render_template('index.html', cities=cities)

@app.route('/results', methods=['POST'])
def results():
    start = request.form['start']
    end = request.form['end']
    k = int(request.form['k'])
    sort_by = int(request.form['sort_by'])

    routes_raw = compute_distances(start, end, graph_dict, k, sort_by)

    # Convert raw routes into structured format
    routes = []
    for r in routes_raw:
        *cities, cost = r  # unpack: all but last = cities; last = cost tuple
        routes.append({
            "cities": cities,
            "cost": cost
        })


    return render_template('results.html', start=start, end=end, routes=routes)


if __name__ == '__main__':
    app.run(debug=True)
