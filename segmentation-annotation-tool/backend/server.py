# your-project/backend/server.py
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure the data directory relative to the server file
MANUSCRIPT_DIR = '/mnt/cai-data/manuscript-annotation-tool/manuscripts/man-seg'
DATA_DIR = os.path.join(MANUSCRIPT_DIR, 'points-2D')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data')


@app.route('/api/points/<int:page>', methods=['GET'])
def get_points(page):
    try:
        filename = f"pg_{page}_points.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Page not found'}), 404
        
        with open(filepath, 'r') as f:
            points = f.read()
            
        return points, 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-labels', methods=['POST'])
def save_labels():
    try:
        data = request.json
        page_num = data.get('page')
        labels = data.get('labels')
        
        if page_num is None or labels is None:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        filename = f"pg_{page_num}_labels.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(map(str, labels)))
        
        return jsonify({'message': f'Successfully saved labels for page {page_num}'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)