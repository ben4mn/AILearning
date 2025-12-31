const express = require('express');
const cors = require('cors');
const config = require('./config/env');
const db = require('./config/database');
const authRoutes = require('./routes/auth.routes');
const errorHandler = require('./middleware/errorHandler');

const app = express();

// Middleware
app.use(cors({
  origin: ['http://localhost:3051', 'http://127.0.0.1:3051'],
  credentials: true,
}));
app.use(express.json());

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    await db.query('SELECT 1');
    res.json({ status: 'healthy', database: 'connected' });
  } catch (error) {
    res.status(503).json({ status: 'unhealthy', database: 'disconnected', error: error.message });
  }
});

// API Routes
app.use('/api/auth', authRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Error handler
app.use(errorHandler);

// Start server
const PORT = config.port;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${config.nodeEnv}`);
});

module.exports = app;
