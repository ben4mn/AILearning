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

// Extended types for API responses

// Topic with lessons (from GET /api/topics/:slug)
export interface TopicWithLessons extends Topic {
  lessons: Lesson[];
  quiz: {
    id: number;
    title: string;
    passingScore: number;
    isGate: boolean;
    questionCount: number;
  } | null;
}

// Topic with metadata for listing (from GET /api/topics/linear)
export interface TopicListItem extends Topic {
  lessonCount: number;
  hasQuiz: boolean;
}

// Topic connection for mind map
export interface TopicConnection {
  id: number;
  fromTopicId: number;
  toTopicId: number;
  fromSlug: string;
  toSlug: string;
  connectionType: 'leads_to' | 'enabled' | 'preceded' | 'conceptual_link' | 'influenced';
  label: string | null;
}

// Lesson with navigation info (from GET /api/lessons/:id/content)
export interface LessonContent {
  lesson: Lesson & {
    topicId: number;
    topicSlug: string;
    topicTitle: string;
  };
  content: string;
  previousLesson: {
    id: number;
    slug: string;
    title: string;
    lessonOrder: number;
  } | null;
  nextLesson: {
    id: number;
    slug: string;
    title: string;
    lessonOrder: number;
  } | null;
}

// Quiz with questions (from GET /api/quiz/:topicSlug)
export interface QuizWithQuestions {
  id: number;
  title: string;
  passingScore: number;
  isGate: boolean;
  questionCount: number;
  questions: Array<{
    id: number;
    questionText: string;
    questionType: 'multiple_choice';
    options: string[];
    questionOrder: number;
  }>;
}

// Quiz submission
export interface QuizSubmission {
  answers: Record<number, string>; // questionId -> selectedAnswer
}

// Quiz result (from POST /api/quiz/:id/submit)
export interface QuizResult {
  quizId: number;
  topicSlug: string;
  score: number;
  passed: boolean;
  passingScore: number;
  correctCount: number;
  totalQuestions: number;
  feedback: Array<{
    questionId: number;
    correct: boolean;
    userAnswer: string | null;
    correctAnswer: string;
    explanation: string;
  }>;
}

// Progress statistics (from GET /api/progress)
export interface ProgressStats {
  lessonsCompleted: number;
  totalLessons: number;
  topicsStarted: number;
  topicsCompleted: number;
  totalTopics: number;
  quizzesPassed: number;
  totalQuizzes: number;
  totalTimeMinutes: number;
}

// Topic progress (from GET /api/progress/topic/:slug)
export interface TopicProgress {
  topicSlug: string;
  lessonsCompleted: number;
  totalLessons: number;
  percentComplete: number;
  lessons: Array<{
    id: number;
    slug: string;
    title: string;
    lessonOrder: number;
    completed: boolean;
    completedAt: string | null;
    timeSpentSeconds: number | null;
  }>;
  quiz: {
    id: number;
    bestScore: number | null;
    passed: boolean;
  } | null;
}

// Progress overview item (from GET /api/progress/overview)
export interface ProgressOverviewItem {
  id: number;
  slug: string;
  title: string;
  era: string;
  linearOrder: number;
  totalLessons: number;
  completedLessons: number;
  percentComplete: number;
  hasQuiz: boolean;
  quizPassed: boolean;
}

// Era definition
export interface Era {
  id: string;
  name: string;
  description: string;
  color: string;
  order: number;
}

// View mode for curriculum navigation
export type ViewMode = 'linear' | 'mindmap';
