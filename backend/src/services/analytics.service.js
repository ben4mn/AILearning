const db = require('../config/database');

const SESSION_TIMEOUT_MINUTES = 30;

/**
 * Start a new session for a user
 * Also closes any stale sessions (inactive for > 30 min)
 */
const startSession = async (userId) => {
  // Close any stale sessions for this user
  await db.query(`
    UPDATE user_sessions
    SET ended_at = last_activity
    WHERE user_id = $1
      AND ended_at IS NULL
      AND last_activity < NOW() - INTERVAL '${SESSION_TIMEOUT_MINUTES} minutes'
  `, [userId]);

  // Create new session
  const result = await db.query(
    `INSERT INTO user_sessions (user_id, started_at, last_activity)
     VALUES ($1, NOW(), NOW())
     RETURNING id, started_at`,
    [userId]
  );

  return {
    sessionId: result.rows[0].id,
    startedAt: result.rows[0].started_at,
  };
};

/**
 * Update session activity (heartbeat)
 * Optionally track a new page visit
 */
const heartbeat = async (sessionId, userId, pagePath = null) => {
  // Update last activity
  const result = await db.query(
    `UPDATE user_sessions
     SET last_activity = NOW()
     WHERE id = $1 AND user_id = $2 AND ended_at IS NULL
     RETURNING id`,
    [sessionId, userId]
  );

  if (result.rows.length === 0) {
    // Session not found or already ended - start a new one
    return startSession(userId);
  }

  // If page path provided and different from last, increment pages visited
  if (pagePath) {
    await db.query(
      `UPDATE user_sessions
       SET pages_visited = pages_visited + 1
       WHERE id = $1`,
      [sessionId]
    );
  }

  return { sessionId, updated: true };
};

/**
 * End a session
 */
const endSession = async (sessionId, userId) => {
  await db.query(
    `UPDATE user_sessions
     SET ended_at = NOW()
     WHERE id = $1 AND user_id = $2 AND ended_at IS NULL`,
    [sessionId, userId]
  );

  return { sessionId, ended: true };
};

/**
 * Log a page view
 */
const logPageView = async (sessionId, userId, pagePath, pageTitle = null) => {
  // Update time_on_page for the previous page in this session
  await db.query(`
    UPDATE page_views
    SET time_on_page_seconds = EXTRACT(EPOCH FROM (NOW() - visited_at))::INT
    WHERE session_id = $1
      AND user_id = $2
      AND time_on_page_seconds IS NULL
      AND id = (
        SELECT id FROM page_views
        WHERE session_id = $1 AND user_id = $2
        ORDER BY visited_at DESC
        LIMIT 1
      )
  `, [sessionId, userId]);

  // Insert new page view
  const result = await db.query(
    `INSERT INTO page_views (session_id, user_id, page_path, page_title, visited_at)
     VALUES ($1, $2, $3, $4, NOW())
     RETURNING id, visited_at`,
    [sessionId, userId, pagePath, pageTitle]
  );

  return {
    pageViewId: result.rows[0].id,
    visitedAt: result.rows[0].visited_at,
  };
};

/**
 * Get active session for a user (if any)
 */
const getActiveSession = async (userId) => {
  const result = await db.query(`
    SELECT id, started_at, last_activity, pages_visited
    FROM user_sessions
    WHERE user_id = $1
      AND ended_at IS NULL
      AND last_activity > NOW() - INTERVAL '${SESSION_TIMEOUT_MINUTES} minutes'
    ORDER BY started_at DESC
    LIMIT 1
  `, [userId]);

  if (result.rows.length === 0) {
    return null;
  }

  return {
    sessionId: result.rows[0].id,
    startedAt: result.rows[0].started_at,
    lastActivity: result.rows[0].last_activity,
    pagesVisited: result.rows[0].pages_visited,
  };
};

module.exports = {
  startSession,
  heartbeat,
  endSession,
  logPageView,
  getActiveSession,
};
