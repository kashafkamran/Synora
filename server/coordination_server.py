from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

# In-memory storage
registered_clients = {}
data_partitions = [
    "dav_swa",  # Kidaw'ida - Kiswahili
    "kln_swa",  # Kalenjin - Kiswahili
    "luo_swa",  # Dholuo - Kiswahili
]

def assign_partition(client_index):
    """Assign a data partition to a client."""
    return data_partitions[client_index % len(data_partitions)]

@app.route('/register', methods=['POST'])
def register_client():
    data = request.get_json()

    # Check if required field exists
    if not data or 'client_name' not in data:
        return jsonify({
            "error": "client_name is required"
        }), 400

    client_name = data['client_name']

    # Check for duplicate registration
    for cid, info in registered_clients.items():
        if info['client_name'] == client_name:
            return jsonify({
                "error": f"Client '{client_name}' is already registered",
                "client_id": cid
            }), 409

    # Generate unique client ID
    client_id = str(uuid.uuid4())

    # Assign data partition
    partition = assign_partition(len(registered_clients))

    # Store client info
    registered_clients[client_id] = {
        "client_name": client_name,
        "partition": partition,
        "status": "active"
    }

    return jsonify({
        "message": "Registration successful",
        "client_id": client_id,
        "partition": partition,
        "status": "active"
    }), 200

@app.route('/clients', methods=['GET'])
def get_clients():
    """View all registered clients."""
    return jsonify({
        "total_clients": len(registered_clients),
        "clients": registered_clients
    }), 200

if __name__ == '__main__':
    print("Coordination Server is running...")
    print("Register at: http://127.0.0.1:5000/register")
    print("View clients at: http://127.0.0.1:5000/clients")
    app.run(debug=True)