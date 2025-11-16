import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GameBoard from './components/GameBoard';
import TrainingInterface from './components/TrainingInterface';
import QTableVisualization from './components/QTableVisualization';
import Analytics from './components/Analytics';
import Header from './components/Header';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('game');

  const tabs = [
    { id: 'game', label: '🎮 Play Game', component: GameBoard },
    { id: 'training', label: '🧠 Training', component: TrainingInterface },
    { id: 'qtable', label: '📊 Q-Table', component: QTableVisualization },
    { id: 'analytics', label: '📈 Analytics', component: Analytics }
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component;

  return (
    <div className="App">
      <Header />
      
      <div className="container">
        {/* Tab Navigation */}
        <motion.div 
          className="tab-navigation"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="tab-content"
          >
            {ActiveComponent && <ActiveComponent />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
