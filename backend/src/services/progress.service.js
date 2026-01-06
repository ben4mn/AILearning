const db = require('../config/database');

/**
 * Get user's overall progress statistics
 */
const getStats = async (userId) => {
  // Get total counts
  const totalsResult = await db.query(`
    SELECT
      (SELECT COUNT(*) FROM topics) as total_topics,
      (SELECT COUNT(*) FROM lessons) as total_lessons,
      (SELECT COUNT(*) FROM quizzes) as total_quizzes
  `);

  const totals = totalsResult.rows[0];

  // Get user's completed counts
  const progressResult = await db.query(`
    SELECT
      (SELECT COUNT(DISTINCT l.topic_id)
       FROM user_progress up
       JOIN lessons l ON l.id = up.lesson_id
       WHERE up.user_id = $1) as topics_touched,
      (SELECT COUNT(*) FROM user_progress WHERE user_id = $1) as lessons_completed,
      (SELECT COUNT(DISTINCT quiz_id) FROM user_quiz_attempts WHERE user_id = $1 AND passed = true) as quizzes_passed,
      (SELECT COALESCE(SUM(time_spent_seconds), 0) FROM user_progress WHERE user_id = $1) as total_time_seconds
  `, [userId]);

  const progress = progressResult.rows[0];

  // Calculate topics completed (all lessons in topic done)
  const topicsCompletedResult = await db.query(`
    SELECT COUNT(*) as completed
    FROM (
      SELECT t.id
      FROM topics t
      WHERE NOT EXISTS (
        SELECT 1 FROM lessons l
        WHERE l.topic_id = t.id
        AND NOT EXISTS (
          SELECT 1 FROM user_progress up
          WHERE up.lesson_id = l.id AND up.user_id = $1
        )
      )
      AND EXISTS (SELECT 1 FROM lessons WHERE topic_id = t.id)
    ) as completed_topics
  `, [userId]);

  return {
    lessonsCompleted: parseInt(progress.lessons_completed, 10),
    totalLessons: parseInt(totals.total_lessons, 10),
    topicsStarted: parseInt(progress.topics_touched, 10),
    topicsCompleted: parseInt(topicsCompletedResult.rows[0].completed, 10),
    totalTopics: parseInt(totals.total_topics, 10),
    quizzesPassed: parseInt(progress.quizzes_passed, 10),
    totalQuizzes: parseInt(totals.total_quizzes, 10),
    totalTimeMinutes: Math.round(parseInt(progress.total_time_seconds, 10) / 60),
  };
};

/**
 * Get user's progress for a specific topic
 */
const getByTopic = async (userId, topicSlug) => {
  // Get topic
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

  // Get all lessons for topic
  const lessonsResult = await db.query(
    `SELECT l.id, l.slug, l.title, l.lesson_order,
            up.completed_at, up.time_spent_seconds
     FROM lessons l
     LEFT JOIN user_progress up ON up.lesson_id = l.id AND up.user_id = $1
     WHERE l.topic_id = $2
     ORDER BY l.lesson_order ASC`,
    [userId, topicId]
  );

  // Get quiz status
  const quizResult = await db.query(
    `SELECT q.id,
            (SELECT MAX(score) FROM user_quiz_attempts WHERE quiz_id = q.id AND user_id = $1) as best_score,
            (SELECT passed FROM user_quiz_attempts WHERE quiz_id = q.id AND user_id = $1 AND passed = true LIMIT 1) as has_passed
     FROM quizzes q
     WHERE q.topic_id = $2`,
    [userId, topicId]
  );

  const completedLessons = lessonsResult.rows.filter(l => l.completed_at).length;
  const totalLessons = lessonsResult.rows.length;

  return {
    topicSlug,
    lessonsCompleted: completedLessons,
    totalLessons,
    percentComplete: totalLessons > 0 ? Math.round((completedLessons / totalLessons) * 100) : 0,
    lessons: lessonsResult.rows.map(l => ({
      id: l.id,
      slug: l.slug,
      title: l.title,
      lessonOrder: l.lesson_order,
      completed: !!l.completed_at,
      completedAt: l.completed_at,
      timeSpentSeconds: l.time_spent_seconds,
    })),
    quiz: quizResult.rows.length > 0 ? {
      id: quizResult.rows[0].id,
      bestScore: quizResult.rows[0].best_score,
      passed: !!quizResult.rows[0].has_passed,
    } : null,
  };
};

/**
 * Mark a lesson as complete
 */
const completeLesson = async (userId, lessonId, timeSpentSeconds = null) => {
  // Check lesson exists
  const lessonResult = await db.query(
    `SELECT l.id, l.title, t.slug as topic_slug
     FROM lessons l
     JOIN topics t ON t.id = l.topic_id
     WHERE l.id = $1`,
    [lessonId]
  );

  if (lessonResult.rows.length === 0) {
    const error = new Error('Lesson not found');
    error.statusCode = 404;
    throw error;
  }

  // Insert or update progress
  await db.query(
    `INSERT INTO user_progress (user_id, lesson_id, completed_at, time_spent_seconds)
     VALUES ($1, $2, NOW(), $3)
     ON CONFLICT (user_id, lesson_id)
     DO UPDATE SET completed_at = NOW(), time_spent_seconds = COALESCE($3, user_progress.time_spent_seconds)`,
    [userId, lessonId, timeSpentSeconds]
  );

  const lesson = lessonResult.rows[0];

  return {
    lessonId: lesson.id,
    lessonTitle: lesson.title,
    topicSlug: lesson.topic_slug,
    completedAt: new Date().toISOString(),
  };
};

/**
 * Get list of completed lesson IDs for a user
 */
const getCompletedLessonIds = async (userId) => {
  const result = await db.query(
    'SELECT lesson_id FROM user_progress WHERE user_id = $1',
    [userId]
  );

  return result.rows.map(r => r.lesson_id);
};

/**
 * Get list of passed quiz IDs for a user
 */
const getPassedQuizIds = async (userId) => {
  const result = await db.query(
    'SELECT DISTINCT quiz_id FROM user_quiz_attempts WHERE user_id = $1 AND passed = true',
    [userId]
  );

  return result.rows.map(r => r.quiz_id);
};

/**
 * Get user's progress overview (for all topics)
 */
const getOverview = async (userId) => {
  const result = await db.query(`
    SELECT
      t.id,
      t.slug,
      t.title,
      t.era,
      t.linear_order,
      COUNT(l.id) as total_lessons,
      COUNT(up.lesson_id) as completed_lessons,
      q.id as quiz_id,
      (SELECT passed FROM user_quiz_attempts WHERE quiz_id = q.id AND user_id = $1 AND passed = true LIMIT 1) as quiz_passed
    FROM topics t
    LEFT JOIN lessons l ON l.topic_id = t.id
    LEFT JOIN user_progress up ON up.lesson_id = l.id AND up.user_id = $1
    LEFT JOIN quizzes q ON q.topic_id = t.id
    GROUP BY t.id, q.id
    ORDER BY t.linear_order ASC
  `, [userId]);

  return result.rows.map(row => ({
    id: row.id,
    slug: row.slug,
    title: row.title,
    era: row.era,
    linearOrder: row.linear_order,
    totalLessons: parseInt(row.total_lessons, 10),
    completedLessons: parseInt(row.completed_lessons, 10),
    percentComplete: row.total_lessons > 0
      ? Math.round((parseInt(row.completed_lessons, 10) / parseInt(row.total_lessons, 10)) * 100)
      : 0,
    hasQuiz: !!row.quiz_id,
    quizPassed: !!row.quiz_passed,
  }));
};

module.exports = {
  getStats,
  getByTopic,
  completeLesson,
  getCompletedLessonIds,
  getPassedQuizIds,
  getOverview,
};
