const db = require('../config/database');

/**
 * Get quiz by topic slug (without correct answers - for taking the quiz)
 */
const getByTopicSlug = async (topicSlug) => {
  // Get topic ID
  const topicResult = await db.query(
    'SELECT id FROM topics WHERE slug = $1',
    [topicSlug]
  );

  if (topicResult.rows.length === 0) {
    const error = new Error('Topic not found');
    error.statusCode = 404;
    throw error;
  }

  const topicId = topicResult.rows[0].id;

  // Get quiz
  const quizResult = await db.query(
    `SELECT id, title, passing_score, is_gate
     FROM quizzes
     WHERE topic_id = $1`,
    [topicId]
  );

  if (quizResult.rows.length === 0) {
    const error = new Error('Quiz not found for this topic');
    error.statusCode = 404;
    throw error;
  }

  const quiz = quizResult.rows[0];

  // Get questions (without correct answers)
  const questionsResult = await db.query(
    `SELECT id, question_text, question_type, options, question_order
     FROM quiz_questions
     WHERE quiz_id = $1
     ORDER BY question_order ASC`,
    [quiz.id]
  );

  return {
    id: quiz.id,
    title: quiz.title,
    passingScore: quiz.passing_score,
    isGate: quiz.is_gate,
    questionCount: questionsResult.rows.length,
    questions: questionsResult.rows.map(q => ({
      id: q.id,
      questionText: q.question_text,
      questionType: q.question_type,
      options: q.options,
      questionOrder: q.question_order,
    })),
  };
};

/**
 * Submit quiz attempt
 * @param {number} quizId - Quiz ID
 * @param {number} userId - User ID
 * @param {Object} answers - Map of questionId -> selectedAnswer
 */
const submit = async (quizId, userId, answers) => {
  // Get quiz with questions and correct answers
  const quizResult = await db.query(
    `SELECT q.id, q.title, q.passing_score, q.topic_id,
            t.slug as topic_slug
     FROM quizzes q
     JOIN topics t ON t.id = q.topic_id
     WHERE q.id = $1`,
    [quizId]
  );

  if (quizResult.rows.length === 0) {
    const error = new Error('Quiz not found');
    error.statusCode = 404;
    throw error;
  }

  const quiz = quizResult.rows[0];

  // Get questions with correct answers
  const questionsResult = await db.query(
    `SELECT id, question_text, correct_answer, explanation
     FROM quiz_questions
     WHERE quiz_id = $1`,
    [quizId]
  );

  const questions = questionsResult.rows;

  // Grade the quiz
  let correctCount = 0;
  const feedback = [];

  for (const question of questions) {
    const userAnswer = answers[question.id];
    const isCorrect = userAnswer === question.correct_answer;

    if (isCorrect) {
      correctCount++;
    }

    feedback.push({
      questionId: question.id,
      correct: isCorrect,
      userAnswer: userAnswer || null,
      correctAnswer: question.correct_answer,
      explanation: question.explanation,
    });
  }

  const score = Math.round((correctCount / questions.length) * 100);
  const passed = score >= quiz.passing_score;

  // Record the attempt
  await db.query(
    `INSERT INTO user_quiz_attempts (user_id, quiz_id, score, passed, answers, attempted_at)
     VALUES ($1, $2, $3, $4, $5, NOW())`,
    [userId, quizId, score, passed, JSON.stringify(answers)]
  );

  return {
    quizId: quiz.id,
    topicSlug: quiz.topic_slug,
    score,
    passed,
    passingScore: quiz.passing_score,
    correctCount,
    totalQuestions: questions.length,
    feedback,
  };
};

/**
 * Get quiz review (with answers) - only after passing or for review
 */
const getReview = async (quizId, userId) => {
  // Check if user has passed this quiz
  const attemptResult = await db.query(
    `SELECT id, score, passed, attempted_at
     FROM user_quiz_attempts
     WHERE user_id = $1 AND quiz_id = $2 AND passed = true
     ORDER BY attempted_at DESC
     LIMIT 1`,
    [userId, quizId]
  );

  if (attemptResult.rows.length === 0) {
    const error = new Error('You must pass the quiz before reviewing answers');
    error.statusCode = 403;
    throw error;
  }

  // Get quiz with all data
  const quizResult = await db.query(
    `SELECT id, title, passing_score
     FROM quizzes
     WHERE id = $1`,
    [quizId]
  );

  if (quizResult.rows.length === 0) {
    const error = new Error('Quiz not found');
    error.statusCode = 404;
    throw error;
  }

  const quiz = quizResult.rows[0];

  // Get questions with correct answers
  const questionsResult = await db.query(
    `SELECT id, question_text, question_type, options, correct_answer, explanation, question_order
     FROM quiz_questions
     WHERE quiz_id = $1
     ORDER BY question_order ASC`,
    [quizId]
  );

  return {
    id: quiz.id,
    title: quiz.title,
    passingScore: quiz.passing_score,
    bestScore: attemptResult.rows[0].score,
    questions: questionsResult.rows.map(q => ({
      id: q.id,
      questionText: q.question_text,
      questionType: q.question_type,
      options: q.options,
      correctAnswer: q.correct_answer,
      explanation: q.explanation,
      questionOrder: q.question_order,
    })),
  };
};

/**
 * Get user's quiz attempts for a specific quiz
 */
const getAttempts = async (quizId, userId) => {
  const result = await db.query(
    `SELECT id, score, passed, attempted_at
     FROM user_quiz_attempts
     WHERE user_id = $1 AND quiz_id = $2
     ORDER BY attempted_at DESC`,
    [userId, quizId]
  );

  return result.rows.map(row => ({
    id: row.id,
    score: row.score,
    passed: row.passed,
    attemptedAt: row.attempted_at,
  }));
};

/**
 * Check if user has passed a quiz
 */
const hasPassed = async (quizId, userId) => {
  const result = await db.query(
    `SELECT id FROM user_quiz_attempts
     WHERE user_id = $1 AND quiz_id = $2 AND passed = true
     LIMIT 1`,
    [userId, quizId]
  );

  return result.rows.length > 0;
};

module.exports = {
  getByTopicSlug,
  submit,
  getReview,
  getAttempts,
  hasPassed,
};
