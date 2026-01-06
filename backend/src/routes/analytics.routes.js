const express = require('express');
const router = express.Router();
const analyticsService = require('../services/analytics.service');
const { authenticateToken } = require('../middleware/auth');

// All analytics routes require authentication
router.use(authenticateToken);

// POST /api/analytics/session/start - Start a new session
router.post('/session/start', async (req, res, next) => {
  try {
    const result = await analyticsService.startSession(req.user.id);
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// POST /api/analytics/session/heartbeat - Update session activity
router.post('/session/heartbeat', async (req, res, next) => {
  try {
    const { sessionId, pagePath } = req.body;

    if (!sessionId) {
      const error = new Error('sessionId is required');
      error.statusCode = 400;
      throw error;
    }

    const result = await analyticsService.heartbeat(
      parseInt(sessionId, 10),
      req.user.id,
      pagePath || null
    );
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// POST /api/analytics/session/end - End a session
router.post('/session/end', async (req, res, next) => {
  try {
    const { sessionId } = req.body;

    if (!sessionId) {
      const error = new Error('sessionId is required');
      error.statusCode = 400;
      throw error;
    }

    const result = await analyticsService.endSession(
      parseInt(sessionId, 10),
      req.user.id
    );
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// POST /api/analytics/pageview - Log a page view
router.post('/pageview', async (req, res, next) => {
  try {
    const { sessionId, pagePath, pageTitle } = req.body;

    if (!sessionId || !pagePath) {
      const error = new Error('sessionId and pagePath are required');
      error.statusCode = 400;
      throw error;
    }

    const result = await analyticsService.logPageView(
      parseInt(sessionId, 10),
      req.user.id,
      pagePath,
      pageTitle || null
    );
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// GET /api/analytics/session/active - Get active session (if any)
router.get('/session/active', async (req, res, next) => {
  try {
    const session = await analyticsService.getActiveSession(req.user.id);
    res.json({ data: session });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
