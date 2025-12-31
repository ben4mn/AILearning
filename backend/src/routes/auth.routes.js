const express = require('express');
const router = express.Router();
const authService = require('../services/auth.service');
const { authenticateToken } = require('../middleware/auth');
const { validateRegistration, validateLogin } = require('../middleware/validate');

// POST /api/auth/register
router.post('/register', validateRegistration, async (req, res, next) => {
  try {
    const { email, password, displayName } = req.body;
    const result = await authService.register({ email, password, displayName });
    res.status(201).json({ data: result });
  } catch (error) {
    next(error);
  }
});

// POST /api/auth/login
router.post('/login', validateLogin, async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const result = await authService.login({ email, password });
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// GET /api/auth/me
router.get('/me', authenticateToken, async (req, res, next) => {
  try {
    const user = await authService.getUser(req.user.id);
    res.json({ data: { user } });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
