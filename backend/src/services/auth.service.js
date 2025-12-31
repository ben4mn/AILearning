const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const db = require('../config/database');
const config = require('../config/env');

const SALT_ROUNDS = 10;
const TOKEN_EXPIRY = '7d';

const register = async ({ email, password, displayName }) => {
  // Check if user already exists
  const existingUser = await db.query(
    'SELECT id FROM users WHERE email = $1',
    [email.toLowerCase()]
  );

  if (existingUser.rows.length > 0) {
    const error = new Error('Email already registered');
    error.statusCode = 409;
    throw error;
  }

  // Hash password
  const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

  // Create user
  const result = await db.query(
    `INSERT INTO users (email, password_hash, display_name, created_at)
     VALUES ($1, $2, $3, NOW())
     RETURNING id, email, display_name, created_at`,
    [email.toLowerCase(), passwordHash, displayName || null]
  );

  const user = result.rows[0];

  // Generate token
  const token = jwt.sign(
    { id: user.id, email: user.email },
    config.jwtSecret,
    { expiresIn: TOKEN_EXPIRY }
  );

  return {
    user: {
      id: user.id,
      email: user.email,
      displayName: user.display_name,
      createdAt: user.created_at,
    },
    token,
  };
};

const login = async ({ email, password }) => {
  // Find user
  const result = await db.query(
    'SELECT id, email, password_hash, display_name, created_at FROM users WHERE email = $1',
    [email.toLowerCase()]
  );

  if (result.rows.length === 0) {
    const error = new Error('Invalid email or password');
    error.statusCode = 401;
    throw error;
  }

  const user = result.rows[0];

  // Verify password
  const isValidPassword = await bcrypt.compare(password, user.password_hash);

  if (!isValidPassword) {
    const error = new Error('Invalid email or password');
    error.statusCode = 401;
    throw error;
  }

  // Update last login
  await db.query(
    'UPDATE users SET last_login = NOW() WHERE id = $1',
    [user.id]
  );

  // Generate token
  const token = jwt.sign(
    { id: user.id, email: user.email },
    config.jwtSecret,
    { expiresIn: TOKEN_EXPIRY }
  );

  return {
    user: {
      id: user.id,
      email: user.email,
      displayName: user.display_name,
      createdAt: user.created_at,
    },
    token,
  };
};

const getUser = async (userId) => {
  const result = await db.query(
    'SELECT id, email, display_name, created_at, last_login FROM users WHERE id = $1',
    [userId]
  );

  if (result.rows.length === 0) {
    const error = new Error('User not found');
    error.statusCode = 404;
    throw error;
  }

  const user = result.rows[0];

  return {
    id: user.id,
    email: user.email,
    displayName: user.display_name,
    createdAt: user.created_at,
    lastLogin: user.last_login,
  };
};

module.exports = {
  register,
  login,
  getUser,
};
