const config = require('../config/env');

const errorHandler = (err, req, res, next) => {
  console.error('Error:', err);

  // Default error
  let statusCode = err.statusCode || 500;
  let message = err.message || 'Internal server error';

  // Handle specific error types
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = err.message;
  }

  if (err.name === 'UnauthorizedError') {
    statusCode = 401;
    message = 'Unauthorized';
  }

  // Don't leak error details in production
  if (config.nodeEnv === 'production' && statusCode === 500) {
    message = 'Internal server error';
  }

  res.status(statusCode).json({
    error: message,
    ...(config.nodeEnv !== 'production' && { stack: err.stack }),
  });
};

module.exports = errorHandler;
