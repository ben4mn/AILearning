const express = require('express');
const router = express.Router();
const progressService = require('../services/progress.service');
const { authenticateToken } = require('../middleware/auth');

// All progress routes require authentication
router.use(authenticateToken);

// GET /api/progress - Get user's overall progress statistics
router.get('/', async (req, res, next) => {
  try {
    const stats = await progressService.getStats(req.user.id);
    res.json({ data: stats });
  } catch (error) {
    next(error);
  }
});

// GET /api/progress/overview - Get progress for all topics
router.get('/overview', async (req, res, next) => {
  try {
    const overview = await progressService.getOverview(req.user.id);
    res.json({ data: overview });
  } catch (error) {
    next(error);
  }
});

// GET /api/progress/completed - Get list of completed lesson IDs
router.get('/completed', async (req, res, next) => {
  try {
    const lessonIds = await progressService.getCompletedLessonIds(req.user.id);
    const quizIds = await progressService.getPassedQuizIds(req.user.id);
    res.json({
      data: {
        completedLessons: lessonIds,
        passedQuizzes: quizIds,
      }
    });
  } catch (error) {
    next(error);
  }
});

// GET /api/progress/topic/:slug - Get progress for specific topic
router.get('/topic/:slug', async (req, res, next) => {
  try {
    const progress = await progressService.getByTopic(req.user.id, req.params.slug);
    res.json({ data: progress });
  } catch (error) {
    next(error);
  }
});

// POST /api/progress/lesson/:id/complete - Mark lesson as complete
router.post('/lesson/:id/complete', async (req, res, next) => {
  try {
    const lessonId = parseInt(req.params.id, 10);
    if (isNaN(lessonId)) {
      const error = new Error('Invalid lesson ID');
      error.statusCode = 400;
      throw error;
    }

    const { timeSpentSeconds } = req.body;
    const result = await progressService.completeLesson(
      req.user.id,
      lessonId,
      timeSpentSeconds || null
    );

    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
