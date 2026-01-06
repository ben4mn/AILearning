const express = require('express');
const router = express.Router();
const quizService = require('../services/quiz.service');
const { authenticateToken } = require('../middleware/auth');

// GET /api/quiz/:topicSlug - Get quiz for a topic (without correct answers)
router.get('/:topicSlug', async (req, res, next) => {
  try {
    const quiz = await quizService.getByTopicSlug(req.params.topicSlug);
    res.json({ data: quiz });
  } catch (error) {
    next(error);
  }
});

// POST /api/quiz/:id/submit - Submit quiz answers (requires auth)
router.post('/:id/submit', authenticateToken, async (req, res, next) => {
  try {
    const quizId = parseInt(req.params.id, 10);
    if (isNaN(quizId)) {
      const error = new Error('Invalid quiz ID');
      error.statusCode = 400;
      throw error;
    }

    const { answers } = req.body;
    if (!answers || typeof answers !== 'object') {
      const error = new Error('Answers are required');
      error.statusCode = 400;
      throw error;
    }

    const result = await quizService.submit(quizId, req.user.id, answers);
    res.json({ data: result });
  } catch (error) {
    next(error);
  }
});

// GET /api/quiz/:id/review - Get quiz with answers (after passing)
router.get('/:id/review', authenticateToken, async (req, res, next) => {
  try {
    const quizId = parseInt(req.params.id, 10);
    if (isNaN(quizId)) {
      const error = new Error('Invalid quiz ID');
      error.statusCode = 400;
      throw error;
    }

    const review = await quizService.getReview(quizId, req.user.id);
    res.json({ data: review });
  } catch (error) {
    next(error);
  }
});

// GET /api/quiz/:id/attempts - Get user's attempts for a quiz
router.get('/:id/attempts', authenticateToken, async (req, res, next) => {
  try {
    const quizId = parseInt(req.params.id, 10);
    if (isNaN(quizId)) {
      const error = new Error('Invalid quiz ID');
      error.statusCode = 400;
      throw error;
    }

    const attempts = await quizService.getAttempts(quizId, req.user.id);
    res.json({ data: attempts });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
