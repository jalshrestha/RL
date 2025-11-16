import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap } from 'lucide-react';
import './Header.css';

const Header = () => {
  return (
    <motion.header 
      className="header"
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      <div className="container">
        <div className="header-content">
          <motion.div 
            className="logo"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Brain className="logo-icon" />
            <div className="logo-text">
              <h1>TicTacToe RL Agent</h1>
              <p>Reinforcement Learning in Action</p>
            </div>
          </motion.div>
          
          <motion.div 
            className="status-indicator"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5, type: "spring", stiffness: 200 }}
          >
            <Zap className="status-icon" />
            <span>AI Powered</span>
          </motion.div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
