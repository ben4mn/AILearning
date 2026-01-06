const db = require('../config/database');

/**
 * Get all topics
 */
const getAll = async () => {
  const result = await db.query(
    `SELECT id, slug, title, description, era, linear_order, icon, estimated_minutes, created_at
     FROM topics
     ORDER BY linear_order ASC`
  );

  return result.rows.map(formatTopic);
};

/**
 * Get topics in linear (chapter) order
 */
const getLinear = async () => {
  const result = await db.query(
    `SELECT t.id, t.slug, t.title, t.description, t.era, t.linear_order, t.icon, t.estimated_minutes,
            COUNT(l.id) as lesson_count,
            q.id as quiz_id
     FROM topics t
     LEFT JOIN lessons l ON l.topic_id = t.id
     LEFT JOIN quizzes q ON q.topic_id = t.id
     GROUP BY t.id, q.id
     ORDER BY t.linear_order ASC`
  );

  return result.rows.map(row => ({
    ...formatTopic(row),
    lessonCount: parseInt(row.lesson_count, 10),
    hasQuiz: !!row.quiz_id,
  }));
};

/**
 * Get single topic by slug with lessons
 */
const getBySlug = async (slug) => {
  // Get topic
  const topicResult = await db.query(
    `SELECT id, slug, title, description, era, linear_order, icon, estimated_minutes, created_at
     FROM topics
     WHERE slug = $1`,
    [slug]
  );

  if (topicResult.rows.length === 0) {
    const error = new Error('Topic not found');
    error.statusCode = 404;
    throw error;
  }

  const topic = formatTopic(topicResult.rows[0]);

  // Get lessons
  const lessonsResult = await db.query(
    `SELECT id, slug, title, content_path, lesson_order, lesson_type
     FROM lessons
     WHERE topic_id = $1
     ORDER BY lesson_order ASC`,
    [topicResult.rows[0].id]
  );

  topic.lessons = lessonsResult.rows.map(formatLesson);

  // Get quiz (without correct answers)
  const quizResult = await db.query(
    `SELECT q.id, q.title, q.passing_score, q.is_gate,
            json_agg(
              json_build_object(
                'id', qq.id,
                'questionText', qq.question_text,
                'questionType', qq.question_type,
                'options', qq.options,
                'questionOrder', qq.question_order
              ) ORDER BY qq.question_order
            ) as questions
     FROM quizzes q
     LEFT JOIN quiz_questions qq ON qq.quiz_id = q.id
     WHERE q.topic_id = $1
     GROUP BY q.id`,
    [topicResult.rows[0].id]
  );

  if (quizResult.rows.length > 0) {
    const quiz = quizResult.rows[0];
    topic.quiz = {
      id: quiz.id,
      title: quiz.title,
      passingScore: quiz.passing_score,
      isGate: quiz.is_gate,
      questionCount: quiz.questions[0]?.id ? quiz.questions.length : 0,
    };
  } else {
    topic.quiz = null;
  }

  return topic;
};

/**
 * Get all topic connections for mind map
 */
const getConnections = async () => {
  const result = await db.query(
    `SELECT tc.id, tc.from_topic_id, tc.to_topic_id, tc.connection_type, tc.label,
            t1.slug as from_slug, t2.slug as to_slug
     FROM topic_connections tc
     JOIN topics t1 ON t1.id = tc.from_topic_id
     JOIN topics t2 ON t2.id = tc.to_topic_id`
  );

  return result.rows.map(row => ({
    id: row.id,
    fromTopicId: row.from_topic_id,
    toTopicId: row.to_topic_id,
    fromSlug: row.from_slug,
    toSlug: row.to_slug,
    connectionType: row.connection_type,
    label: row.label,
  }));
};

/**
 * Get topics grouped by era
 */
const getByEra = async () => {
  const topics = await getLinear();

  // Group by era
  const byEra = {};
  for (const topic of topics) {
    if (!byEra[topic.era]) {
      byEra[topic.era] = [];
    }
    byEra[topic.era].push(topic);
  }

  return byEra;
};

/**
 * Format topic row from database
 */
function formatTopic(row) {
  return {
    id: row.id,
    slug: row.slug,
    title: row.title,
    description: row.description,
    era: row.era,
    linearOrder: row.linear_order,
    icon: row.icon,
    estimatedMinutes: row.estimated_minutes,
    createdAt: row.created_at,
  };
}

/**
 * Format lesson row from database
 */
function formatLesson(row) {
  return {
    id: row.id,
    slug: row.slug,
    title: row.title,
    contentPath: row.content_path,
    lessonOrder: row.lesson_order,
    lessonType: row.lesson_type,
  };
}

module.exports = {
  getAll,
  getLinear,
  getBySlug,
  getConnections,
  getByEra,
};
