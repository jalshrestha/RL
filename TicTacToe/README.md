# TicTacToe Reinforcement Learning Agent

A full-stack web application featuring a Q-learning AI agent that learns to play TicTacToe through reinforcement learning. The application includes a modern React frontend with real-time training visualization, game analytics, and Q-table inspection.

## 🚀 Features

### 🎮 Interactive Gameplay
- Play against the trained RL agent
- Choose who starts first (human or agent)
- Beautiful animated game board
- Real-time game state updates
- Game history tracking

### 🧠 Training Interface
- Real-time training progress tracking
- Customizable hyperparameters:
  - Learning rate (α)
  - Discount factor (γ)
  - Exploration rate (ε)
  - Epsilon decay
- Visual progress indicators
- Start/stop training controls
- Q-table persistence (save/load)

### 📊 Q-Table Visualization
- Interactive Q-table browser
- State-action value visualization
- Search and filter capabilities
- Color-coded Q-values
- Export functionality
- Detailed state inspection

### 📈 Analytics Dashboard
- Game outcome statistics
- Training progress charts
- Agent performance metrics
- Q-table analysis
- Learning insights
- Recent games history

## 🛠️ Technology Stack

### Backend
- **Flask**: Python web framework
- **Flask-CORS**: Cross-origin resource sharing
- **NumPy**: Numerical computing
- **Pickle**: Object serialization

### Frontend
- **React 18**: Modern UI library
- **Framer Motion**: Smooth animations
- **Recharts**: Data visualization
- **Axios**: HTTP client
- **Lucide React**: Beautiful icons

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```
   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

## 🎯 How to Use

### 1. Training the Agent
1. Navigate to the **Training** tab
2. Adjust hyperparameters as needed:
   - **Episodes**: Number of training games (recommended: 1,000,000)
   - **Learning Rate**: How fast the agent learns (0.1)
   - **Discount Factor**: Importance of future rewards (0.9)
   - **Epsilon**: Initial exploration rate (1.0)
   - **Epsilon Decay**: Rate of exploration decrease (0.99995)
   - **Min Epsilon**: Minimum exploration rate (0.01)
3. Click **Start Training**
4. Monitor progress in real-time
5. Save the trained Q-table when complete

### 2. Playing Against the Agent
1. Navigate to the **Play Game** tab
2. Choose who starts first:
   - **You Start**: Human plays as X
   - **Agent Starts**: Agent plays as X
3. Click on empty cells to make your move
4. Watch the agent's strategic responses
5. Play multiple games to see the agent's skill

### 3. Exploring the Q-Table
1. Navigate to the **Q-Table** tab
2. Browse learned states and their Q-values
3. Use search to find specific board positions
4. Click on states to see detailed action values
5. Export data for further analysis

### 4. Viewing Analytics
1. Navigate to the **Analytics** tab
2. View comprehensive performance metrics
3. Analyze training progress charts
4. Review game outcome statistics
5. Get insights into agent learning

## 🧠 Reinforcement Learning Details

### Q-Learning Algorithm
The agent uses Q-learning with the following update rule:

```
Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate
- **γ (gamma)**: Discount factor
- **r**: Immediate reward
- **s'**: Next state
- **max_a'Q(s',a')**: Maximum Q-value for next state

### Reward Structure
- **Win**: +1 reward
- **Loss**: -1 reward
- **Draw**: +0.5 reward
- **Continue**: 0 reward

### State Representation
- Board states are represented as 9-character strings
- Agent uses perspective normalization (always sees itself as 'X')
- Empty cells: ' ' (space)
- Player symbols: 'X' and 'O'

### Exploration Strategy
- **Epsilon-greedy**: Random exploration vs. optimal exploitation
- **Decay**: Exploration decreases over time
- **Minimum**: Maintains small exploration rate

## 📁 Project Structure

```
TicTacToe/
├── app.py                          # Flask backend API
├── tictacagent.py                  # RL agent implementation
├── requirements.txt                 # Python dependencies
├── q_table.pkl                     # Saved Q-table (generated)
├── frontend/                        # React frontend
│   ├── package.json                # Node.js dependencies
│   ├── public/
│   │   ├── index.html              # HTML template
│   │   └── manifest.json           # PWA manifest
│   └── src/
│       ├── index.js                # React entry point
│       ├── App.js                  # Main app component
│       ├── App.css                 # App styles
│       ├── index.css               # Global styles
│       └── components/              # React components
│           ├── Header.js           # App header
│           ├── Header.css
│           ├── GameBoard.js         # Game interface
│           ├── GameBoard.css
│           ├── TrainingInterface.js # Training controls
│           ├── TrainingInterface.css
│           ├── QTableVisualization.js # Q-table browser
│           ├── QTableVisualization.css
│           ├── Analytics.js         # Analytics dashboard
│           └── Analytics.css
└── README.md                        # This file
```

## 🔧 API Endpoints

### Game Management
- `POST /api/game/new` - Start new game
- `POST /api/game/move` - Make a move
- `GET /api/game/status` - Get game status

### Training
- `POST /api/training/start` - Start training
- `GET /api/training/status` - Get training progress
- `POST /api/training/stop` - Stop training

### Q-Table Management
- `POST /api/qtable/save` - Save Q-table
- `POST /api/qtable/load` - Load Q-table
- `GET /api/qtable/stats` - Get Q-table statistics
- `GET /api/qtable/visualize` - Get visualization data

### Analytics
- `GET /api/analytics/game-history` - Get game history
- `GET /api/analytics/agent-performance` - Get performance metrics

## 🎨 Customization

### Styling
- Modify CSS files in `frontend/src/components/`
- Global styles in `frontend/src/index.css`
- Uses CSS custom properties for easy theming

### Hyperparameters
- Adjust default values in `TrainingInterface.js`
- Modify training parameters in `app.py`
- Experiment with different reward structures

### UI Components
- Add new components in `frontend/src/components/`
- Extend existing components for new features
- Use Framer Motion for animations

## 🚀 Deployment

### Backend Deployment
1. Use a WSGI server like Gunicorn
2. Configure environment variables
3. Set up proper CORS settings
4. Deploy to platforms like Heroku, AWS, or DigitalOcean

### Frontend Deployment
1. Build the production version:
   ```bash
   cd frontend
   npm run build
   ```
2. Serve the `build` folder with a web server
3. Deploy to platforms like Netlify, Vercel, or AWS S3

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Q-learning algorithm implementation
- React and Flask communities
- Framer Motion for smooth animations
- Recharts for data visualization

---

**Happy Learning! 🧠🎮**
