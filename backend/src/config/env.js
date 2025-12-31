require('dotenv').config();

module.exports = {
  port: process.env.PORT || 3050,
  databaseUrl: process.env.DATABASE_URL || 'postgresql://aihistory:aihistory_secure_password@localhost:5433/aihistory',
  jwtSecret: process.env.JWT_SECRET || 'dev_jwt_secret_key',
  nodeEnv: process.env.NODE_ENV || 'development',
};
