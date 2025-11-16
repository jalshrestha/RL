"""
Flask Backend API for TicTacToe RL Agent Web Application
======================================================

This Flask application provides REST API endpoints for:
- Training the RL agent
- Playing games against the agent
- Managing Q-table persistence
- Real-time game state updates
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import threading
import time
from tictacagent import (
    TicTacToe, get_Q, choose_action, update_Q, train, save_Q, load_Q,
    get_state_from_perspective, Q
)

app = Flask(__name__)
CORS(app)

# Global variables for game state and training
current_game = None
training_progress = {"status": "idle", "progress": 0, "episodes": 0, "epsilon": 1.0}
training_thread = None

@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    return send_from_directory('frontend/build', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('frontend/build/static', path)

# Game Management Endpoints
@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Start a new game"""
    global current_game
    data = request.get_json()
    starter = data.get('starter', 'human')  # 'human' or 'agent'
    
    current_game = {
        'env': TicTacToe(),
        'starter': starter,
        'human_symbol': 'X' if starter == 'human' else 'O',
        'agent_symbol': 'O' if starter == 'human' else 'X',
        'status': 'active',
        'winner': None,
        'move_history': []
    }
    
    # If agent starts, make first move
    if starter == 'agent':
        make_agent_move()
    
    return jsonify({
        'board': current_game['env'].board,
        'current_player': current_game['agent_symbol'] if starter == 'agent' else current_game['human_symbol'],
        'status': current_game['status'],
        'winner': current_game['winner']
    })

@app.route('/api/game/move', methods=['POST'])
def make_move():
    """Make a human move"""
    global current_game
    
    if not current_game or current_game['status'] != 'active':
        return jsonify({'error': 'No active game'}), 400
    
    data = request.get_json()
    position = data.get('position')
    
    if position is None or position < 0 or position > 8:
        return jsonify({'error': 'Invalid position'}), 400
    
    # Make human move
    success = current_game['env'].make_move(position, current_game['human_symbol'])
    
    if not success:
        return jsonify({'error': 'Position already occupied'}), 400
    
    # Record move
    current_game['move_history'].append({
        'player': current_game['human_symbol'],
        'position': position,
        'timestamp': time.time()
    })
    
    # Check for game end
    check_game_end()
    
    # If game continues and it's agent's turn, make agent move
    if current_game['status'] == 'active':
        make_agent_move()
    
    return jsonify({
        'board': current_game['env'].board,
        'current_player': current_game['agent_symbol'] if current_game['status'] == 'active' else None,
        'status': current_game['status'],
        'winner': current_game['winner'],
        'move_history': current_game['move_history']
    })

def make_agent_move():
    """Make agent move (internal function)"""
    global current_game
    
    if not current_game or current_game['status'] != 'active':
        return
    
    available_actions = current_game['env'].available_actions()
    if not available_actions:
        return
    
    # Get state from agent's perspective
    state_perspective = get_state_from_perspective(
        current_game['env'].board, 
        current_game['agent_symbol']
    )
    
    # Choose action (pure exploitation during gameplay)
    action = choose_action(state_perspective, available_actions, 0.0)
    
    # Make the move
    current_game['env'].make_move(action, current_game['agent_symbol'])
    
    # Record move
    current_game['move_history'].append({
        'player': current_game['agent_symbol'],
        'position': action,
        'timestamp': time.time()
    })
    
    # Check for game end
    check_game_end()

def check_game_end():
    """Check if game has ended"""
    global current_game
    
    if current_game['env'].current_winner:
        current_game['winner'] = current_game['env'].current_winner
        current_game['status'] = 'finished'
    elif current_game['env'].is_full():
        current_game['winner'] = 'draw'
        current_game['status'] = 'finished'

@app.route('/api/game/status', methods=['GET'])
def get_game_status():
    """Get current game status"""
    global current_game
    
    if not current_game:
        return jsonify({'error': 'No active game'}), 400
    
    return jsonify({
        'board': current_game['env'].board,
        'status': current_game['status'],
        'winner': current_game['winner'],
        'move_history': current_game['move_history'],
        'human_symbol': current_game['human_symbol'],
        'agent_symbol': current_game['agent_symbol']
    })

# Training Endpoints
@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training the RL agent"""
    global training_thread, training_progress
    
    if training_thread and training_thread.is_alive():
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.get_json()
    episodes = data.get('episodes', 100000)
    alpha = data.get('alpha', 0.1)
    gamma = data.get('gamma', 0.9)
    epsilon = data.get('epsilon', 1.0)
    epsilon_decay = data.get('epsilon_decay', 0.99995)
    epsilon_min = data.get('epsilon_min', 0.01)
    
    # Reset training progress
    training_progress = {
        "status": "training",
        "progress": 0,
        "episodes": episodes,
        "epsilon": epsilon,
        "current_episode": 0
    }
    
    # Start training in separate thread
    training_thread = threading.Thread(
        target=train_with_progress,
        args=(episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'message': 'Training started', 'episodes': episodes})

def train_with_progress(episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
    """Training function with progress updates"""
    global training_progress
    
    env = TicTacToe()
    
    for ep in range(episodes):
        # Start a new game
        env.reset()
        player = 'X'
        history = []
        
        while True:
            available = env.available_actions()
            state_perspective = get_state_from_perspective(env.board, player)
            action = choose_action(state_perspective, available, epsilon)
            history.append((player, state_perspective, action))
            env.make_move(action, player)
            
            if env.current_winner == player:
                for p, s, a in history:
                    if p == player:
                        reward = 1
                        update_Q(s, a, reward, None, alpha, gamma, [])
                    else:
                        reward = -1
                        update_Q(s, a, reward, None, alpha, gamma, [])
                break
            elif env.is_full():
                for p, s, a in history:
                    reward = 0.5
                    update_Q(s, a, reward, None, alpha, gamma, [])
                break
            else:
                player = 'O' if player == 'X' else 'X'
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update progress
        training_progress.update({
            "progress": (ep + 1) / episodes * 100,
            "current_episode": ep + 1,
            "epsilon": epsilon
        })
    
    training_progress["status"] = "completed"
    save_Q()

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training progress"""
    return jsonify(training_progress)

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training (if possible)"""
    global training_thread, training_progress
    
    if training_thread and training_thread.is_alive():
        training_progress["status"] = "stopping"
        return jsonify({'message': 'Training stop requested'})
    
    return jsonify({'error': 'No training in progress'}), 400

# Q-table Management
@app.route('/api/qtable/save', methods=['POST'])
def save_qtable():
    """Save Q-table to file"""
    try:
        save_Q()
        return jsonify({'message': 'Q-table saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qtable/load', methods=['POST'])
def load_qtable():
    """Load Q-table from file"""
    try:
        load_Q()
        return jsonify({'message': 'Q-table loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/qtable/stats', methods=['GET'])
def get_qtable_stats():
    """Get Q-table statistics"""
    global Q
    
    if not Q:
        return jsonify({'error': 'No Q-table loaded'}), 400
    
    states = set(key[0] for key in Q.keys())
    actions = set(key[1] for key in Q.keys())
    
    return jsonify({
        'total_entries': len(Q),
        'unique_states': len(states),
        'unique_actions': len(actions),
        'avg_q_value': sum(Q.values()) / len(Q) if Q else 0,
        'max_q_value': max(Q.values()) if Q else 0,
        'min_q_value': min(Q.values()) if Q else 0
    })

@app.route('/api/qtable/visualize', methods=['GET'])
def visualize_qtable():
    """Get Q-table visualization data"""
    global Q
    
    if not Q:
        return jsonify({'error': 'No Q-table loaded'}), 400
    
    # Group by state for visualization
    state_actions = {}
    for (state, action), q_value in Q.items():
        if state not in state_actions:
            state_actions[state] = {}
        state_actions[state][action] = q_value
    
    return jsonify({
        'state_actions': state_actions,
        'total_states': len(state_actions)
    })

# Analytics Endpoints
@app.route('/api/analytics/game-history', methods=['GET'])
def get_game_history():
    """Get game history for analytics"""
    # This would typically come from a database
    # For now, return mock data
    return jsonify({
        'games_played': 0,
        'human_wins': 0,
        'agent_wins': 0,
        'draws': 0,
        'win_rate': 0
    })

@app.route('/api/analytics/agent-performance', methods=['GET'])
def get_agent_performance():
    """Get agent performance metrics"""
    global Q
    
    if not Q:
        return jsonify({'error': 'No Q-table loaded'}), 400
    
    # Calculate some performance metrics
    total_q_values = list(Q.values())
    positive_q_values = [q for q in total_q_values if q > 0]
    
    return jsonify({
        'total_learned_states': len(set(key[0] for key in Q.keys())),
        'total_q_entries': len(Q),
        'positive_q_values': len(positive_q_values),
        'confidence_score': len(positive_q_values) / len(total_q_values) if total_q_values else 0,
        'avg_q_value': sum(total_q_values) / len(total_q_values) if total_q_values else 0
    })

if __name__ == '__main__':
    # Load existing Q-table if available
    try:
        load_Q()
        print("✅ Loaded existing Q-table")
    except:
        print("⚠️ No existing Q-table found")
    
    app.run(debug=False, host='0.0.0.0', port=5001)
