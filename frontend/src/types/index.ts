// User types
export interface User {
  id: number;
  email: string;
  displayName: string | null;
  createdAt: string;
  lastLogin?: string;
}

// Auth types
export interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterCredentials {
  email: string;
  password: string;
  displayName?: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
}

export interface ApiError {
  error: string;
}

// Topic types
export interface Topic {
  id: number;
  slug: string;
  title: string;
  description: string | null;
  era: string | null;
  linearOrder: number | null;
  icon: string | null;
  estimatedMinutes: number;
}

// Lesson types
export interface Lesson {
  id: number;
  topicId: number;
  slug: string;
  title: string;
  contentPath: string | null;
  lessonOrder: number | null;
  lessonType: 'content' | 'interactive' | 'video';
}

// Progress types
export interface UserProgress {
  lessonId: number;
  completedAt: string;
  timeSpentSeconds: number | null;
}

// Quiz types
export interface Quiz {
  id: number;
  topicId: number;
  title: string;
  passingScore: number;
  isGate: boolean;
}

export interface QuizQuestion {
  id: number;
  quizId: number;
  questionText: string;
  questionType: 'multiple_choice';
  options: string[];
  explanation: string | null;
  questionOrder: number | null;
}

export interface QuizAttempt {
  quizId: number;
  score: number;
  passed: boolean;
  attemptedAt: string;
}
