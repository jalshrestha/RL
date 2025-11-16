import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { TrendingUp, Trophy, Target, Brain, Bot, RefreshCw } from 'lucide-react';
import axios from 'axios';
import './Analytics.css';

const Analytics = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [timeRange, setTimeRange] = useState('all');

  useEffect(() => {
    loadAnalyticsData();
  }, []);

  const loadAnalyticsData = async () => {
    setIsLoading(true);
    try {
      // Mock data loading - replace with actual API calls when backend is ready
      console.log('Loading analytics data...');
    } catch (error) {
      console.error('Error loading analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Mock data for demonstration (replace with real data from backend)
  const mockGameHistory = {
    games_played: 156,
    human_wins: 23,
    agent_wins: 98,
    draws: 35,
    win_rate: 0.147
  };

  const mockPerformanceData = {
    total_learned_states: 4521,
    total_q_entries: 12847,
    positive_q_values: 8934,
    confidence_score: 0.695,
    avg_q_value: 0.234
  };

  const mockTrainingProgress = [
    { episode: 0, epsilon: 1.0, avg_reward: 0 },
    { episode: 10000, epsilon: 0.95, avg_reward: 0.1 },
    { episode: 50000, epsilon: 0.8, avg_reward: 0.3 },
    { episode: 100000, epsilon: 0.6, avg_reward: 0.5 },
    { episode: 200000, epsilon: 0.4, avg_reward: 0.7 },
    { episode: 500000, epsilon: 0.2, avg_reward: 0.8 },
    { episode: 1000000, epsilon: 0.01, avg_reward: 0.85 }
  ];

  const gameOutcomeData = [
    { name: 'Agent Wins', value: mockGameHistory.agent_wins, color: '#3b82f6' },
    { name: 'Human Wins', value: mockGameHistory.human_wins, color: '#10b981' },
    { name: 'Draws', value: mockGameHistory.draws, color: '#f59e0b' }
  ];

  const performanceMetrics = [
    {
      title: 'Games Played',
      value: mockGameHistory.games_played,
      icon: <Trophy className="metric-icon" />,
      color: '#3b82f6',
      change: '+12%'
    },
    {
      title: 'Agent Win Rate',
      value: `${(mockGameHistory.agent_wins / mockGameHistory.games_played * 100).toFixed(1)}%`,
      icon: <Bot className="metric-icon" />,
      color: '#10b981',
      change: '+5.2%'
    },
    {
      title: 'Learned States',
      value: mockPerformanceData.total_learned_states.toLocaleString(),
      icon: <Brain className="metric-icon" />,
      color: '#8b5cf6',
      change: '+8.1%'
    },
    {
      title: 'Confidence Score',
      value: `${(mockPerformanceData.confidence_score * 100).toFixed(1)}%`,
      icon: <Target className="metric-icon" />,
      color: '#f59e0b',
      change: '+2.3%'
    }
  ];

  const recentGamesData = [
    { game: 1, result: 'Agent Win', moves: 7, duration: '2m 15s' },
    { game: 2, result: 'Draw', moves: 9, duration: '3m 42s' },
    { game: 3, result: 'Human Win', moves: 6, duration: '1m 58s' },
    { game: 4, result: 'Agent Win', moves: 8, duration: '2m 33s' },
    { game: 5, result: 'Agent Win', moves: 5, duration: '1m 45s' }
  ];

  return (
    <div className="analytics">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <h2>📈 Analytics Dashboard</h2>
          <div className="header-controls">
            <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
              <option value="all">All Time</option>
              <option value="week">Last Week</option>
              <option value="month">Last Month</option>
            </select>
            <button className="btn btn-secondary" onClick={loadAnalyticsData} disabled={isLoading}>
              <RefreshCw className={`btn-icon ${isLoading ? 'spinning' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        {performanceMetrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            className="metric-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="metric-header">
              <div className="metric-icon-container" style={{ backgroundColor: metric.color }}>
                {metric.icon}
              </div>
              <div className="metric-change" style={{ color: metric.color }}>
                {metric.change}
              </div>
            </div>
            <div className="metric-content">
              <div className="metric-value">{metric.value}</div>
              <div className="metric-title">{metric.title}</div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="charts-row">
        {/* Game Outcomes Pie Chart */}
        <div className="card chart-card">
          <h3>🎯 Game Outcomes</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={gameOutcomeData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {gameOutcomeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-legend">
            {gameOutcomeData.map((item, index) => (
              <div key={index} className="legend-item">
                <div className="legend-color" style={{ backgroundColor: item.color }}></div>
                <span>{item.name}: {item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Training Progress Line Chart */}
        <div className="card chart-card">
          <h3>📈 Training Progress</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={mockTrainingProgress}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="avg_reward" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-description">
            Average reward per episode during training
          </div>
        </div>
      </div>

      {/* Q-Table Statistics */}
      <div className="card">
        <h3>🧠 Q-Table Analysis</h3>
        <div className="qtable-analysis">
          <div className="analysis-grid">
            <div className="analysis-item">
              <div className="analysis-label">Total Q-Values</div>
              <div className="analysis-value">{mockPerformanceData.total_q_entries.toLocaleString()}</div>
            </div>
            <div className="analysis-item">
              <div className="analysis-label">Positive Q-Values</div>
              <div className="analysis-value">{mockPerformanceData.positive_q_values.toLocaleString()}</div>
            </div>
            <div className="analysis-item">
              <div className="analysis-label">Average Q-Value</div>
              <div className="analysis-value">{mockPerformanceData.avg_q_value.toFixed(3)}</div>
            </div>
            <div className="analysis-item">
              <div className="analysis-label">Learning Efficiency</div>
              <div className="analysis-value">
                {((mockPerformanceData.positive_q_values / mockPerformanceData.total_q_entries) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          
          <div className="progress-bars">
            <div className="progress-item">
              <div className="progress-label">Exploration vs Exploitation</div>
              <div className="progress-bar-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill exploration" 
                    style={{ width: '15%' }}
                  ></div>
                  <div 
                    className="progress-fill exploitation" 
                    style={{ width: '85%' }}
                  ></div>
                </div>
                <div className="progress-labels">
                  <span>15% Exploration</span>
                  <span>85% Exploitation</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Games */}
      <div className="card">
        <h3>🎮 Recent Games</h3>
        <div className="recent-games">
          <div className="games-header">
            <div>Game</div>
            <div>Result</div>
            <div>Moves</div>
            <div>Duration</div>
          </div>
          {recentGamesData.map((game, index) => (
            <motion.div
              key={index}
              className="game-row"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="game-number">#{game.game}</div>
              <div className={`game-result ${game.result.toLowerCase().replace(' ', '-')}`}>
                {game.result}
              </div>
              <div className="game-moves">{game.moves}</div>
              <div className="game-duration">{game.duration}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Performance Insights */}
      <div className="card">
        <h3>💡 Performance Insights</h3>
        <div className="insights">
          <div className="insight-item">
            <TrendingUp className="insight-icon" />
            <div className="insight-content">
              <div className="insight-title">Agent is Learning</div>
              <div className="insight-description">
                The agent's win rate has improved by 15% over the last 50 games, 
                showing effective learning from experience.
              </div>
            </div>
          </div>
          
          <div className="insight-item">
            <Brain className="insight-icon" />
            <div className="insight-content">
              <div className="insight-title">Q-Table Growth</div>
              <div className="insight-description">
                The agent has learned {mockPerformanceData.total_learned_states.toLocaleString()} unique states, 
                indicating comprehensive game understanding.
              </div>
            </div>
          </div>
          
          <div className="insight-item">
            <Target className="insight-icon" />
            <div className="insight-content">
              <div className="insight-title">Optimal Strategy</div>
              <div className="insight-description">
                With a confidence score of {(mockPerformanceData.confidence_score * 100).toFixed(1)}%, 
                the agent demonstrates strong strategic decision-making.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
