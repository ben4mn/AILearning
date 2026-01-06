const express = require('express');
const router = express.Router();
const lessonsService = require('../services/lessons.service');

// GET /api/lessons/:id - Get lesson metadata
router.get('/:id', async (req, res, next) => {
  try {
    const lessonId = parseInt(req.params.id, 10);
    if (isNaN(lessonId)) {
      const error = new Error('Invalid lesson ID');
      error.statusCode = 400;
      throw error;
    }

    const lesson = await lessonsService.getById(lessonId);
    res.json({ data: lesson });
  } catch (error) {
    next(error);
  }
});

// GET /api/lessons/:id/content - Get lesson with markdown content
router.get('/:id/content', async (req, res, next) => {
  try {
    const lessonId = parseInt(req.params.id, 10);
    if (isNaN(lessonId)) {
      const error = new Error('Invalid lesson ID');
      error.statusCode = 400;
      throw error;
    }

    const lessonContent = await lessonsService.getContent(lessonId);
    res.json({ data: lessonContent });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
