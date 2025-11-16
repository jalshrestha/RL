import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, Eye, EyeOff, Download, RefreshCw } from 'lucide-react';
import axios from 'axios';
import './QTableVisualization.css';

const QTableVisualization = () => {
  const [qtableData, setQtableData] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('q_value');
  const [sortOrder, setSortOrder] = useState('desc');
  const [showEmptyStates, setShowEmptyStates] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedState, setSelectedState] = useState(null);

  useEffect(() => {
    loadQtableData();
  }, []);

  useEffect(() => {
    if (qtableData) {
      filterAndSortData();
    }
  }, [qtableData, searchTerm, sortBy, sortOrder, showEmptyStates]);

  const loadQtableData = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/qtable/visualize');
      setQtableData(response.data);
    } catch (error) {
      console.error('Error loading Q-table data:', error);
      alert('Failed to load Q-table data. Please train the agent first.');
    } finally {
      setIsLoading(false);
    }
  };

  const filterAndSortData = () => {
    let filtered = Object.entries(qtableData.state_actions);
    
    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(([state]) => 
        state.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Filter empty states
    if (!showEmptyStates) {
      filtered = filtered.filter(([state]) => 
        state.includes('X') || state.includes('O')
      );
    }
    
    // Sort data
    filtered.sort((a, b) => {
      const [stateA, actionsA] = a;
      const [stateB, actionsB] = b;
      
      let valueA, valueB;
      
      switch (sortBy) {
        case 'q_value':
          valueA = Math.max(...Object.values(actionsA));
          valueB = Math.max(...Object.values(actionsB));
          break;
        case 'state':
          valueA = stateA;
          valueB = stateB;
          break;
        case 'actions':
          valueA = Object.keys(actionsA).length;
          valueB = Object.keys(actionsB).length;
          break;
        default:
          valueA = Math.max(...Object.values(actionsA));
          valueB = Math.max(...Object.values(actionsB));
      }
      
      if (sortOrder === 'asc') {
        return valueA > valueB ? 1 : -1;
      } else {
        return valueA < valueB ? 1 : -1;
      }
    });
    
    setFilteredData(filtered);
  };

  const formatState = (state) => {
    // Convert state string to visual representation
    const chars = state.split('');
    return chars.map((char, index) => (
      <span key={index} className={`state-cell ${char === ' ' ? 'empty' : char.toLowerCase()}`}>
        {char === ' ' ? '·' : char}
      </span>
    ));
  };

  const getQValueColor = (value) => {
    if (value > 0.5) return '#10b981'; // Green for high positive values
    if (value > 0) return '#3b82f6';   // Blue for positive values
    if (value > -0.5) return '#f59e0b'; // Yellow for slightly negative
    return '#ef4444'; // Red for very negative values
  };

  const exportQtable = () => {
    if (!qtableData) return;
    
    const csvContent = [
      'State,Q-Values',
      ...Object.entries(qtableData.state_actions).map(([state, actions]) => 
        `${state},"${JSON.stringify(actions)}"`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'qtable_export.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getActionPosition = (action) => {
    const positions = [
      [0, 0], [0, 1], [0, 2],
      [1, 0], [1, 1], [1, 2],
      [2, 0], [2, 1], [2, 2]
    ];
    return positions[action] || [0, 0];
  };

  return (
    <div className="qtable-visualization">
      {/* Controls */}
      <div className="card">
        <div className="card-header">
          <h2>📊 Q-Table Visualization</h2>
          <div className="header-controls">
            <button className="btn btn-secondary" onClick={loadQtableData} disabled={isLoading}>
              <RefreshCw className={`btn-icon ${isLoading ? 'spinning' : ''}`} />
              Refresh
            </button>
            <button className="btn btn-success" onClick={exportQtable} disabled={!qtableData}>
              <Download className="btn-icon" />
              Export CSV
            </button>
          </div>
        </div>
        
        <div className="controls">
          <div className="search-controls">
            <div className="search-box">
              <Search className="search-icon" />
              <input
                type="text"
                placeholder="Search states..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="filter-controls">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={showEmptyStates}
                  onChange={(e) => setShowEmptyStates(e.target.checked)}
                />
                <span>Show empty states</span>
              </label>
            </div>
          </div>
          
          <div className="sort-controls">
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="q_value">Sort by Q-Value</option>
              <option value="state">Sort by State</option>
              <option value="actions">Sort by Actions</option>
            </select>
            
            <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value)}>
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>
        </div>
      </div>

      {/* Q-Table Data */}
      {isLoading ? (
        <div className="card loading-card">
          <div className="loading-container">
            <div className="loading"></div>
            <span>Loading Q-table data...</span>
          </div>
        </div>
      ) : filteredData && filteredData.length > 0 ? (
        <div className="card">
          <div className="qtable-header">
            <h3>Q-Table Entries ({filteredData.length} states)</h3>
            <div className="legend">
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#10b981' }}></div>
                <span>High Positive (>0.5)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#3b82f6' }}></div>
                <span>Positive (0-0.5)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#f59e0b' }}></div>
                <span>Low Negative (-0.5-0)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#ef4444' }}></div>
                <span>High Negative (&lt;-0.5)</span>
              </div>
            </div>
          </div>
          
          <div className="qtable-grid">
            {filteredData.slice(0, 100).map(([state, actions], index) => (
              <motion.div
                key={state}
                className="qtable-item"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.01 }}
                onClick={() => setSelectedState(selectedState === state ? null : state)}
              >
                <div className="state-visualization">
                  <div className="state-board">
                    {formatState(state)}
                  </div>
                </div>
                
                <div className="actions-container">
                  {Object.entries(actions).map(([action, qValue]) => {
                    const [row, col] = getActionPosition(parseInt(action));
                    return (
                      <div
                        key={action}
                        className="action-item"
                        style={{ backgroundColor: getQValueColor(qValue) }}
                        title={`Action ${action}: Q-value = ${qValue.toFixed(3)}`}
                      >
                        <div className="action-position">{action}</div>
                        <div className="q-value">{qValue.toFixed(2)}</div>
                      </div>
                    );
                  })}
                </div>
                
                <div className="state-summary">
                  <div className="max-q">
                    Max Q: {Math.max(...Object.values(actions)).toFixed(3)}
                  </div>
                  <div className="action-count">
                    {Object.keys(actions).length} actions
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          
          {filteredData.length > 100 && (
            <div className="load-more">
              <p>Showing first 100 states. Use search/filter to narrow down results.</p>
            </div>
          )}
        </div>
      ) : (
        <div className="card empty-state">
          <div className="empty-content">
            <EyeOff className="empty-icon" />
            <h3>No Q-table data available</h3>
            <p>Train the agent first to see Q-table visualization.</p>
            <button className="btn btn-primary" onClick={() => window.location.hash = '#training'}>
              Go to Training
            </button>
          </div>
        </div>
      )}

      {/* State Detail Modal */}
      <AnimatePresence>
        {selectedState && qtableData && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedState(null)}
          >
            <motion.div
              className="modal-content"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h3>State Details</h3>
                <button onClick={() => setSelectedState(null)}>×</button>
              </div>
              
              <div className="modal-body">
                <div className="state-board-large">
                  {formatState(selectedState)}
                </div>
                
                <div className="actions-detail">
                  <h4>Available Actions & Q-Values</h4>
                  {Object.entries(qtableData.state_actions[selectedState]).map(([action, qValue]) => {
                    const [row, col] = getActionPosition(parseInt(action));
                    return (
                      <div key={action} className="action-detail-item">
                        <div className="action-info">
                          <span className="action-number">Action {action}</span>
                          <span className="action-position">Position ({row}, {col})</span>
                        </div>
                        <div 
                          className="q-value-large"
                          style={{ backgroundColor: getQValueColor(qValue) }}
                        >
                          {qValue.toFixed(4)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default QTableVisualization;
