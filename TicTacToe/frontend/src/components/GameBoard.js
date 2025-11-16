import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, User, Bot, Trophy } from 'lucide-react';
import axios from 'axios';
import './GameBoard.css';

const GameBoard = () => {
  const [gameState, setGameState] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [gameHistory, setGameHistory] = useState([]);
  const [showGameSetup, setShowGameSetup] = useState(true);

  const startNewGame = async (starter) => {
    setIsLoading(true);
    try {
      const response = await axios.post('/api/game/new', { starter });
      setGameState(response.data);
      setShowGameSetup(false);
    } catch (error) {
      console.error('Error starting game:', error);
      alert('Failed to start game. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const makeMove = async (position) => {
    if (!gameState || gameState.status !== 'active' || isLoading) return;
    
    setIsLoading(true);
    try {
      const response = await axios.post('/api/game/move', { position });
      setGameState(response.data);
      
      // Add to game history
      if (response.data.status === 'finished') {
        setGameHistory(prev => [...prev, {
          winner: response.data.winner,
          moves: response.data.move_history,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Error making move:', error);
      alert('Invalid move. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const resetGame = () => {
    setGameState(null);
    setShowGameSetup(true);
  };

  const getCurrentPlayer = () => {
    if (!gameState) return null;
    const xCount = gameState.board.filter(cell => cell === 'X').length;
    const oCount = gameState.board.filter(cell => cell === 'O').length;
    return xCount === oCount ? 'X' : 'O';
  };

  const getGameStatusMessage = () => {
    if (!gameState) return '';
    
    if (gameState.status === 'finished') {
      if (gameState.winner === 'draw') return "It's a draw! 🤝";
      if (gameState.winner === gameState.human_symbol) return "You win! 🎉";
      return "Agent wins! 🤖";
    }
    
    const currentPlayer = getCurrentPlayer();
    if (currentPlayer === gameState.human_symbol) {
      return "Your turn! 👤";
    }
    return "Agent is thinking... 🤖";
  };


  const isWinningCell = (index) => {
    if (!gameState || !gameState.winner || gameState.winner === 'draw') return false;
    
    // Simple winning line detection (you could enhance this)
    const board = gameState.board;
    const winner = gameState.winner;
    
    // Check rows
    for (let row = 0; row < 3; row++) {
      if (board[row * 3] === winner && board[row * 3 + 1] === winner && board[row * 3 + 2] === winner) {
        return index >= row * 3 && index < (row + 1) * 3;
      }
    }
    
    // Check columns
    for (let col = 0; col < 3; col++) {
      if (board[col] === winner && board[col + 3] === winner && board[col + 6] === winner) {
        return index % 3 === col;
      }
    }
    
    // Check diagonals
    if (board[0] === winner && board[4] === winner && board[8] === winner) {
      return index === 0 || index === 4 || index === 8;
    }
    if (board[2] === winner && board[4] === winner && board[6] === winner) {
      return index === 2 || index === 4 || index === 6;
    }
    
    return false;
  };

  return (
    <div className="game-board-container">
      {showGameSetup ? (
        <motion.div 
          className="game-setup"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="card">
            <h2>🎮 Start New Game</h2>
            <p>Choose who should start the game:</p>
            
            <div className="starter-options">
              <motion.button
                className="btn btn-primary starter-btn"
                onClick={() => startNewGame('human')}
                disabled={isLoading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <User className="btn-icon" />
                <div>
                  <div>You Start</div>
                  <small>You play as X</small>
                </div>
              </motion.button>
              
              <motion.button
                className="btn btn-secondary starter-btn"
                onClick={() => startNewGame('agent')}
                disabled={isLoading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Bot className="btn-icon" />
                <div>
                  <div>Agent Starts</div>
                  <small>Agent plays as X</small>
                </div>
              </motion.button>
            </div>
            
            {isLoading && (
              <div className="loading-container">
                <div className="loading"></div>
                <span>Starting game...</span>
              </div>
            )}
          </div>
        </motion.div>
      ) : (
        <motion.div 
          className="game-play"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Game Status */}
          <div className="card game-status">
            <div className="status-header">
              <h3>{getGameStatusMessage()}</h3>
              <button className="btn btn-secondary" onClick={resetGame}>
                <RotateCcw className="btn-icon" />
                New Game
              </button>
            </div>
            
            {gameState && (
              <div className="game-info">
                <div className="player-info">
                  <div className="player">
                    <User className="player-icon" />
                    <span>You: {gameState.human_symbol}</span>
                  </div>
                  <div className="player">
                    <Bot className="player-icon" />
                    <span>Agent: {gameState.agent_symbol}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Game Board */}
          <div className="card">
            <div className="board-container">
              <div className="board">
                {gameState?.board.map((cell, index) => (
                  <motion.button
                    key={index}
                    className={`cell ${isWinningCell(index) ? 'winning' : ''}`}
                    onClick={() => makeMove(index)}
                    disabled={cell !== ' ' || isLoading || gameState.status !== 'active'}
                    whileHover={{ scale: cell === ' ' ? 1.05 : 1 }}
                    whileTap={{ scale: cell === ' ' ? 0.95 : 1 }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <AnimatePresence>
                      {cell !== ' ' && (
                        <motion.span
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          exit={{ scale: 0, rotate: 180 }}
                          transition={{ type: "spring", stiffness: 200 }}
                          className={`symbol ${cell}`}
                        >
                          {cell}
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </motion.button>
                ))}
              </div>
            </div>
          </div>

          {/* Game History */}
          {gameHistory.length > 0 && (
            <div className="card">
              <h3>🏆 Recent Games</h3>
              <div className="game-history">
                {gameHistory.slice(-5).reverse().map((game, index) => (
                  <motion.div
                    key={index}
                    className="history-item"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <Trophy className="trophy-icon" />
                    <div>
                      <div className="result">
                        {game.winner === 'draw' ? 'Draw' : 
                         game.winner === gameState?.human_symbol ? 'You Won' : 'Agent Won'}
                      </div>
                      <small>{game.timestamp.toLocaleTimeString()}</small>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default GameBoard;
