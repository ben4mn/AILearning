const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

const validatePassword = (password) => {
  return password && password.length >= 6;
};

const validateRegistration = (req, res, next) => {
  const { email, password, displayName } = req.body;

  const errors = [];

  if (!email) {
    errors.push('Email is required');
  } else if (!validateEmail(email)) {
    errors.push('Invalid email format');
  }

  if (!password) {
    errors.push('Password is required');
  } else if (!validatePassword(password)) {
    errors.push('Password must be at least 6 characters');
  }

  if (displayName && displayName.length > 100) {
    errors.push('Display name must be less than 100 characters');
  }

  if (errors.length > 0) {
    return res.status(400).json({ error: errors.join(', ') });
  }

  next();
};

const validateLogin = (req, res, next) => {
  const { email, password } = req.body;

  const errors = [];

  if (!email) {
    errors.push('Email is required');
  }

  if (!password) {
    errors.push('Password is required');
  }

  if (errors.length > 0) {
    return res.status(400).json({ error: errors.join(', ') });
  }

  next();
};

module.exports = {
  validateRegistration,
  validateLogin,
};
