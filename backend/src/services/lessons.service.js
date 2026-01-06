const fs = require('fs').promises;
const path = require('path');
const db = require('../config/database');

// Content directory path - use env var or default to /app/content for Docker
const CONTENT_DIR = process.env.CONTENT_DIR || '/app/content';

/**
 * Get lesson by ID
 */
const getById = async (lessonId) => {
  const result = await db.query(
    `SELECT l.id, l.slug, l.title, l.content_path, l.lesson_order, l.lesson_type,
            l.topic_id, t.slug as topic_slug, t.title as topic_title
     FROM lessons l
     JOIN topics t ON t.id = l.topic_id
     WHERE l.id = $1`,
    [lessonId]
  );

  if (result.rows.length === 0) {
    const error = new Error('Lesson not found');
    error.statusCode = 404;
    throw error;
  }

  return formatLesson(result.rows[0]);
};

/**
 * Get lesson content (reads markdown file)
 */
const getContent = async (lessonId) => {
  const lesson = await getById(lessonId);

  // Read markdown content
  let content = '';
  if (lesson.contentPath) {
    const filePath = path.join(CONTENT_DIR, lesson.contentPath);
    try {
      content = await fs.readFile(filePath, 'utf-8');
    } catch (err) {
      if (err.code === 'ENOENT') {
        // File doesn't exist yet - return placeholder
        content = `# ${lesson.title}\n\n*Content coming soon...*\n`;
      } else {
        throw err;
      }
    }
  }

  // Get prev/next lessons
  const navResult = await db.query(
    `SELECT id, slug, title, lesson_order
     FROM lessons
     WHERE topic_id = $1
     ORDER BY lesson_order ASC`,
    [lesson.topicId]
  );

  const lessons = navResult.rows;
  const currentIndex = lessons.findIndex(l => l.id === lesson.id);

  const previousLesson = currentIndex > 0
    ? formatNavLesson(lessons[currentIndex - 1])
    : null;

  const nextLesson = currentIndex < lessons.length - 1
    ? formatNavLesson(lessons[currentIndex + 1])
    : null;

  return {
    lesson,
    content,
    previousLesson,
    nextLesson,
  };
};

/**
 * Get all lessons for a topic
 */
const getByTopicId = async (topicId) => {
  const result = await db.query(
    `SELECT id, slug, title, content_path, lesson_order, lesson_type
     FROM lessons
     WHERE topic_id = $1
     ORDER BY lesson_order ASC`,
    [topicId]
  );

  return result.rows.map(row => ({
    id: row.id,
    slug: row.slug,
    title: row.title,
    contentPath: row.content_path,
    lessonOrder: row.lesson_order,
    lessonType: row.lesson_type,
  }));
};

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
    topicId: row.topic_id,
    topicSlug: row.topic_slug,
    topicTitle: row.topic_title,
  };
}

/**
 * Format lesson for navigation (prev/next)
 */
function formatNavLesson(row) {
  return {
    id: row.id,
    slug: row.slug,
    title: row.title,
    lessonOrder: row.lesson_order,
  };
}

module.exports = {
  getById,
  getContent,
  getByTopicId,
};
