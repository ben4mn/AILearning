/**
 * Database Seed Script
 *
 * Reads curriculum.json and populates the database with topics, lessons, quizzes, and connections.
 *
 * Usage: node src/db/seed.js [--clear]
 *   --clear: Clear existing data before seeding (default in development)
 */

const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../.env') });

const db = require('../config/database');
const curriculum = require('../data/curriculum.json');

const CLEAR_DATA = process.argv.includes('--clear') || process.env.NODE_ENV !== 'production';

async function clearExistingData() {
  console.log('Clearing existing content data...');

  // Delete in order to respect foreign key constraints
  await db.query('DELETE FROM user_quiz_attempts');
  await db.query('DELETE FROM user_progress');
  await db.query('DELETE FROM quiz_questions');
  await db.query('DELETE FROM quizzes');
  await db.query('DELETE FROM lessons');
  await db.query('DELETE FROM topic_connections');
  await db.query('DELETE FROM topics');
  await db.query('DELETE FROM achievements');

  console.log('Existing data cleared.');
}

async function seedTopics() {
  console.log(`Seeding ${curriculum.topics.length} topics...`);

  const topicIdMap = {}; // slug -> id

  for (const topic of curriculum.topics) {
    const result = await db.query(
      `INSERT INTO topics (slug, title, description, era, linear_order, icon, estimated_minutes)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        topic.slug,
        topic.title,
        topic.description,
        topic.era,
        topic.linearOrder,
        topic.icon,
        topic.estimatedMinutes
      ]
    );

    topicIdMap[topic.slug] = result.rows[0].id;
    console.log(`  ✓ Topic: ${topic.title}`);
  }

  return topicIdMap;
}

async function seedLessons(topicIdMap) {
  console.log('Seeding lessons...');

  let lessonCount = 0;

  for (const topic of curriculum.topics) {
    const topicId = topicIdMap[topic.slug];

    for (const lesson of topic.lessons) {
      await db.query(
        `INSERT INTO lessons (topic_id, slug, title, content_path, lesson_order, lesson_type)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        [
          topicId,
          lesson.slug,
          lesson.title,
          lesson.contentPath,
          lesson.lessonOrder,
          lesson.lessonType
        ]
      );
      lessonCount++;
    }
  }

  console.log(`  ✓ ${lessonCount} lessons seeded`);
}

async function seedConnections(topicIdMap) {
  console.log('Seeding topic connections...');

  let connectionCount = 0;

  for (const connection of curriculum.connections) {
    const fromId = topicIdMap[connection.fromTopicSlug];
    const toId = topicIdMap[connection.toTopicSlug];

    if (!fromId || !toId) {
      console.warn(`  ⚠ Skipping connection: ${connection.fromTopicSlug} -> ${connection.toTopicSlug} (topic not found)`);
      continue;
    }

    await db.query(
      `INSERT INTO topic_connections (from_topic_id, to_topic_id, connection_type, label)
       VALUES ($1, $2, $3, $4)`,
      [fromId, toId, connection.connectionType, connection.label]
    );
    connectionCount++;
  }

  console.log(`  ✓ ${connectionCount} connections seeded`);
}

async function seedQuizzes(topicIdMap) {
  console.log('Seeding quizzes and questions...');

  let quizCount = 0;
  let questionCount = 0;

  for (const topic of curriculum.topics) {
    if (!topic.quiz) continue;

    const topicId = topicIdMap[topic.slug];
    const quiz = topic.quiz;

    // Insert quiz
    const quizResult = await db.query(
      `INSERT INTO quizzes (topic_id, title, passing_score, is_gate)
       VALUES ($1, $2, $3, $4)
       RETURNING id`,
      [topicId, quiz.title, quiz.passingScore, quiz.isGate]
    );

    const quizId = quizResult.rows[0].id;
    quizCount++;

    // Insert questions
    for (const question of quiz.questions) {
      await db.query(
        `INSERT INTO quiz_questions (quiz_id, question_text, question_type, options, correct_answer, explanation, question_order)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [
          quizId,
          question.questionText,
          question.questionType,
          JSON.stringify(question.options),
          question.correctAnswer,
          question.explanation,
          question.questionOrder
        ]
      );
      questionCount++;
    }
  }

  console.log(`  ✓ ${quizCount} quizzes with ${questionCount} questions seeded`);
}

async function seedAchievements() {
  console.log('Seeding achievements...');

  const achievements = [
    {
      slug: 'first-steps',
      title: 'First Steps',
      description: 'Complete your first lesson',
      icon: 'footprints',
      criteria: { type: 'lessons_completed', count: 1 }
    },
    {
      slug: 'quick-learner',
      title: 'Quick Learner',
      description: 'Complete 5 lessons',
      icon: 'zap',
      criteria: { type: 'lessons_completed', count: 5 }
    },
    {
      slug: 'knowledge-seeker',
      title: 'Knowledge Seeker',
      description: 'Complete 10 lessons',
      icon: 'book-open',
      criteria: { type: 'lessons_completed', count: 10 }
    },
    {
      slug: 'quiz-taker',
      title: 'Quiz Taker',
      description: 'Pass your first quiz',
      icon: 'check-circle',
      criteria: { type: 'quizzes_passed', count: 1 }
    },
    {
      slug: 'quiz-master',
      title: 'Quiz Master',
      description: 'Score 100% on any quiz',
      icon: 'trophy',
      criteria: { type: 'quiz_perfect_score', count: 1 }
    },
    {
      slug: 'era-explorer',
      title: 'Era Explorer',
      description: 'Complete all topics in any era',
      icon: 'compass',
      criteria: { type: 'era_completed', count: 1 }
    },
    {
      slug: 'time-traveler',
      title: 'Time Traveler',
      description: 'Complete topics from three different eras',
      icon: 'clock',
      criteria: { type: 'eras_touched', count: 3 }
    },
    {
      slug: 'ai-historian',
      title: 'AI Historian',
      description: 'Complete all topics in the curriculum',
      icon: 'award',
      criteria: { type: 'all_topics_completed' }
    }
  ];

  for (const achievement of achievements) {
    await db.query(
      `INSERT INTO achievements (slug, title, description, icon, criteria)
       VALUES ($1, $2, $3, $4, $5)`,
      [
        achievement.slug,
        achievement.title,
        achievement.description,
        achievement.icon,
        JSON.stringify(achievement.criteria)
      ]
    );
  }

  console.log(`  ✓ ${achievements.length} achievements seeded`);
}

async function seed() {
  console.log('\n🌱 Starting database seed...\n');
  console.log(`Curriculum: ${curriculum.metadata.title}`);
  console.log(`Topics: ${curriculum.topics.length}`);
  console.log(`Connections: ${curriculum.connections.length}`);
  console.log('');

  try {
    if (CLEAR_DATA) {
      await clearExistingData();
      console.log('');
    }

    const topicIdMap = await seedTopics();
    await seedLessons(topicIdMap);
    await seedConnections(topicIdMap);
    await seedQuizzes(topicIdMap);
    await seedAchievements();

    console.log('\n✅ Database seed completed successfully!\n');

    // Print summary
    const topicCount = await db.query('SELECT COUNT(*) FROM topics');
    const lessonCount = await db.query('SELECT COUNT(*) FROM lessons');
    const quizCount = await db.query('SELECT COUNT(*) FROM quizzes');
    const questionCount = await db.query('SELECT COUNT(*) FROM quiz_questions');

    console.log('Summary:');
    console.log(`  Topics: ${topicCount.rows[0].count}`);
    console.log(`  Lessons: ${lessonCount.rows[0].count}`);
    console.log(`  Quizzes: ${quizCount.rows[0].count}`);
    console.log(`  Questions: ${questionCount.rows[0].count}`);

  } catch (error) {
    console.error('\n❌ Seed failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  } finally {
    await db.pool.end();
  }
}

seed();
