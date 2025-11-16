import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, Brain, TrendingUp, Zap, Save, Download } from 'lucide-react';
import axios from 'axios';
import './TrainingInterface.css';

const TrainingInterface = () => {
  const [trainingParams, setTrainingParams] = useState({
    episodes: 100000,
    alpha: 0.1,
    gamma: 0.9,
    epsilon: 1.0,
    epsilon_decay: 0.99995,
    epsilon_min: 0.01
  });
  
  const [trainingStatus, setTrainingStatus] = useState({
    status: 'idle',
    progress: 0,
    episodes: 0,
    epsilon: 1.0,
    current_episode: 0
  });
  
  const [isTraining, setIsTraining] = useState(false);
  const [qtableStats, setQtableStats] = useState(null);

  useEffect(() => {
    // Poll training status
    const interval = setInterval(async () => {
      if (isTraining) {
        try {
          const response = await axios.get('/api/training/status');
          setTrainingStatus(response.data);
          
          if (response.data.status === 'completed') {
            setIsTraining(false);
            fetchQtableStats();
          }
        } catch (error) {
          console.error('Error fetching training status:', error);
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isTraining]);

  useEffect(() => {
    fetchQtableStats();
  }, []);

  const fetchQtableStats = async () => {
    try {
      const response = await axios.get('/api/qtable/stats');
      setQtableStats(response.data);
    } catch (error) {
      console.error('Error fetching Q-table stats:', error);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    try {
      await axios.post('/api/training/start', trainingParams);
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training. Please try again.');
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('/api/training/stop');
      setIsTraining(false);
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const saveQtable = async () => {
    try {
      await axios.post('/api/qtable/save');
      alert('Q-table saved successfully!');
    } catch (error) {
      console.error('Error saving Q-table:', error);
      alert('Failed to save Q-table.');
    }
  };

  const loadQtable = async () => {
    try {
      await axios.post('/api/qtable/load');
      alert('Q-table loaded successfully!');
      fetchQtableStats();
    } catch (error) {
      console.error('Error loading Q-table:', error);
      alert('Failed to load Q-table.');
    }
  };

  const resetParams = () => {
    setTrainingParams({
      episodes: 100000,
      alpha: 0.1,
      gamma: 0.9,
      epsilon: 1.0,
      epsilon_decay: 0.99995,
      epsilon_min: 0.01
    });
  };

  const getStatusColor = () => {
    switch (trainingStatus.status) {
      case 'training': return '#10b981';
      case 'completed': return '#3b82f6';
      case 'stopping': return '#f59e0b';
      default: return '#64748b';
    }
  };

  const getStatusText = () => {
    switch (trainingStatus.status) {
      case 'training': return 'Training in Progress';
      case 'completed': return 'Training Completed';
      case 'stopping': return 'Stopping Training';
      default: return 'Ready to Train';
    }
  };

  return (
    <div className="training-interface">
      {/* Training Parameters */}
      <div className="card">
        <div className="card-header">
          <h2>🧠 Training Parameters</h2>
          <button className="btn btn-secondary" onClick={resetParams}>
            <RotateCcw className="btn-icon" />
            Reset
          </button>
        </div>
        
        <div className="params-grid">
          <div className="param-group">
            <label>Episodes</label>
            <input
              type="number"
              value={trainingParams.episodes}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, episodes: parseInt(e.target.value) || 0 }))}
              disabled={isTraining}
              min="1000"
              max="10000000"
            />
            <small>Number of training games</small>
          </div>
          
          <div className="param-group">
            <label>Learning Rate (α)</label>
            <input
              type="number"
              step="0.01"
              value={trainingParams.alpha}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, alpha: parseFloat(e.target.value) || 0 }))}
              disabled={isTraining}
              min="0.01"
              max="1.0"
            />
            <small>How fast the agent learns</small>
          </div>
          
          <div className="param-group">
            <label>Discount Factor (γ)</label>
            <input
              type="number"
              step="0.01"
              value={trainingParams.gamma}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, gamma: parseFloat(e.target.value) || 0 }))}
              disabled={isTraining}
              min="0.1"
              max="1.0"
            />
            <small>Importance of future rewards</small>
          </div>
          
          <div className="param-group">
            <label>Initial Epsilon</label>
            <input
              type="number"
              step="0.01"
              value={trainingParams.epsilon}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, epsilon: parseFloat(e.target.value) || 0 }))}
              disabled={isTraining}
              min="0.1"
              max="1.0"
            />
            <small>Initial exploration rate</small>
          </div>
          
          <div className="param-group">
            <label>Epsilon Decay</label>
            <input
              type="number"
              step="0.00001"
              value={trainingParams.epsilon_decay}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, epsilon_decay: parseFloat(e.target.value) || 0 }))}
              disabled={isTraining}
              min="0.9"
              max="0.99999"
            />
            <small>Rate of exploration decrease</small>
          </div>
          
          <div className="param-group">
            <label>Min Epsilon</label>
            <input
              type="number"
              step="0.01"
              value={trainingParams.epsilon_min}
              onChange={(e) => setTrainingParams(prev => ({ ...prev, epsilon_min: parseFloat(e.target.value) || 0 }))}
              disabled={isTraining}
              min="0.01"
              max="0.1"
            />
            <small>Minimum exploration rate</small>
          </div>
        </div>
      </div>

      {/* Training Controls */}
      <div className="card">
        <div className="card-header">
          <h2>🎮 Training Controls</h2>
          <div className="status-indicator" style={{ color: getStatusColor() }}>
            <div className="status-dot" style={{ backgroundColor: getStatusColor() }}></div>
            {getStatusText()}
          </div>
        </div>
        
        <div className="training-controls">
          <motion.button
            className={`btn ${isTraining ? 'btn-danger' : 'btn-success'}`}
            onClick={isTraining ? stopTraining : startTraining}
            disabled={trainingStatus.status === 'stopping'}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isTraining ? (
              <>
                <Pause className="btn-icon" />
                Stop Training
              </>
            ) : (
              <>
                <Play className="btn-icon" />
                Start Training
              </>
            )}
          </motion.button>
          
          <div className="control-buttons">
            <button className="btn btn-secondary" onClick={saveQtable}>
              <Save className="btn-icon" />
              Save Q-Table
            </button>
            <button className="btn btn-secondary" onClick={loadQtable}>
              <Download className="btn-icon" />
              Load Q-Table
            </button>
          </div>
        </div>
      </div>

      {/* Training Progress */}
      <AnimatePresence>
        {isTraining && (
          <motion.div 
            className="card"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h3>📊 Training Progress</h3>
            
            <div className="progress-container">
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${trainingStatus.progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <div className="progress-text">
                {trainingStatus.current_episode.toLocaleString()} / {trainingStatus.episodes.toLocaleString()} episodes
                ({trainingStatus.progress.toFixed(1)}%)
              </div>
            </div>
            
            <div className="training-stats">
              <div className="stat">
                <Brain className="stat-icon" />
                <div>
                  <div className="stat-value">{trainingStatus.epsilon.toFixed(4)}</div>
                  <div className="stat-label">Current Epsilon</div>
                </div>
              </div>
              
              <div className="stat">
                <TrendingUp className="stat-icon" />
                <div>
                  <div className="stat-value">{((1 - trainingStatus.epsilon) * 100).toFixed(1)}%</div>
                  <div className="stat-label">Exploitation Rate</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Q-Table Statistics */}
      {qtableStats && (
        <div className="card">
          <h3>📈 Q-Table Statistics</h3>
          
          <div className="stats-grid">
            <div className="stat-card">
              <Zap className="stat-icon" />
              <div className="stat-content">
                <div className="stat-value">{qtableStats.total_entries.toLocaleString()}</div>
                <div className="stat-label">Total Q-Values</div>
              </div>
            </div>
            
            <div className="stat-card">
              <Brain className="stat-icon" />
              <div className="stat-content">
                <div className="stat-value">{qtableStats.unique_states.toLocaleString()}</div>
                <div className="stat-label">Unique States</div>
              </div>
            </div>
            
            <div className="stat-card">
              <TrendingUp className="stat-icon" />
              <div className="stat-content">
                <div className="stat-value">{qtableStats.avg_q_value.toFixed(3)}</div>
                <div className="stat-label">Average Q-Value</div>
              </div>
            </div>
            
            <div className="stat-card">
              <Zap className="stat-icon" />
              <div className="stat-content">
                <div className="stat-value">{qtableStats.max_q_value.toFixed(3)}</div>
                <div className="stat-label">Max Q-Value</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingInterface;
