const express = require('express');
const router = express.Router();
const topicsService = require('../services/topics.service');
const { optionalAuth } = require('../middleware/auth');

// GET /api/topics - Get all topics
router.get('/', async (req, res, next) => {
  try {
    const topics = await topicsService.getAll();
    res.json({ data: topics });
  } catch (error) {
    next(error);
  }
});

// GET /api/topics/linear - Get topics in chapter order with metadata
router.get('/linear', async (req, res, next) => {
  try {
    const topics = await topicsService.getLinear();
    res.json({ data: topics });
  } catch (error) {
    next(error);
  }
});

// GET /api/topics/by-era - Get topics grouped by era
router.get('/by-era', async (req, res, next) => {
  try {
    const topicsByEra = await topicsService.getByEra();
    res.json({ data: topicsByEra });
  } catch (error) {
    next(error);
  }
});

// GET /api/topics/connections - Get all topic connections (for mind map)
router.get('/connections', async (req, res, next) => {
  try {
    const connections = await topicsService.getConnections();
    res.json({ data: connections });
  } catch (error) {
    next(error);
  }
});

// GET /api/topics/:slug - Get single topic with lessons
router.get('/:slug', async (req, res, next) => {
  try {
    const topic = await topicsService.getBySlug(req.params.slug);
    res.json({ data: topic });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
