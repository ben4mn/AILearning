import axios, { AxiosError } from 'axios';
import type {
  ApiResponse,
  ApiError,
  AuthResponse,
  LoginCredentials,
  RegisterCredentials,
  User,
  Topic,
  TopicListItem,
  TopicWithLessons,
  TopicConnection,
  LessonContent,
  QuizWithQuestions,
  QuizResult,
  QuizSubmission,
  ProgressStats,
  TopicProgress,
  ProgressOverviewItem,
} from '@/types';

// In production, use relative URL to go through nginx proxy
// In development (with Vite), use localhost:3050 directly
const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:3050' : '');

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle response errors
client.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    if (error.response?.status === 401) {
      // Clear token on unauthorized
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  register: async (credentials: RegisterCredentials): Promise<AuthResponse> => {
    const response = await client.post<ApiResponse<AuthResponse>>('/api/auth/register', credentials);
    return response.data.data;
  },

  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await client.post<ApiResponse<AuthResponse>>('/api/auth/login', credentials);
    return response.data.data;
  },

  getMe: async (): Promise<User> => {
    const response = await client.get<ApiResponse<{ user: User }>>('/api/auth/me');
    return response.data.data.user;
  },
};

// Topics API
export const topicsApi = {
  getAll: async (): Promise<Topic[]> => {
    const response = await client.get<ApiResponse<Topic[]>>('/api/topics');
    return response.data.data;
  },

  getLinear: async (): Promise<TopicListItem[]> => {
    const response = await client.get<ApiResponse<TopicListItem[]>>('/api/topics/linear');
    return response.data.data;
  },

  getBySlug: async (slug: string): Promise<TopicWithLessons> => {
    const response = await client.get<ApiResponse<TopicWithLessons>>(`/api/topics/${slug}`);
    return response.data.data;
  },

  getConnections: async (): Promise<TopicConnection[]> => {
    const response = await client.get<ApiResponse<TopicConnection[]>>('/api/topics/connections');
    return response.data.data;
  },

  getByEra: async (): Promise<Record<string, TopicListItem[]>> => {
    const response = await client.get<ApiResponse<Record<string, TopicListItem[]>>>('/api/topics/by-era');
    return response.data.data;
  },
};

// Lessons API
export const lessonsApi = {
  getContent: async (lessonId: number): Promise<LessonContent> => {
    const response = await client.get<ApiResponse<LessonContent>>(`/api/lessons/${lessonId}/content`);
    return response.data.data;
  },
};

// Quiz API
export const quizApi = {
  getByTopicSlug: async (topicSlug: string): Promise<QuizWithQuestions> => {
    const response = await client.get<ApiResponse<QuizWithQuestions>>(`/api/quiz/${topicSlug}`);
    return response.data.data;
  },

  submit: async (quizId: number, submission: QuizSubmission): Promise<QuizResult> => {
    const response = await client.post<ApiResponse<QuizResult>>(`/api/quiz/${quizId}/submit`, submission);
    return response.data.data;
  },

  getReview: async (quizId: number): Promise<QuizWithQuestions & { bestScore: number }> => {
    const response = await client.get<ApiResponse<QuizWithQuestions & { bestScore: number }>>(`/api/quiz/${quizId}/review`);
    return response.data.data;
  },
};

// Progress API
export const progressApi = {
  getStats: async (): Promise<ProgressStats> => {
    const response = await client.get<ApiResponse<ProgressStats>>('/api/progress');
    return response.data.data;
  },

  getOverview: async (): Promise<ProgressOverviewItem[]> => {
    const response = await client.get<ApiResponse<ProgressOverviewItem[]>>('/api/progress/overview');
    return response.data.data;
  },

  getByTopic: async (topicSlug: string): Promise<TopicProgress> => {
    const response = await client.get<ApiResponse<TopicProgress>>(`/api/progress/topic/${topicSlug}`);
    return response.data.data;
  },

  completeLesson: async (lessonId: number, timeSpentSeconds?: number): Promise<void> => {
    await client.post(`/api/progress/lesson/${lessonId}/complete`, { timeSpentSeconds });
  },

  getCompleted: async (): Promise<{ completedLessons: number[]; passedQuizzes: number[] }> => {
    const response = await client.get<ApiResponse<{ completedLessons: number[]; passedQuizzes: number[] }>>('/api/progress/completed');
    return response.data.data;
  },
};

// Analytics API
export interface SessionResponse {
  sessionId: number;
  startedAt?: string;
  updated?: boolean;
  ended?: boolean;
}

export interface PageViewResponse {
  pageViewId: number;
  visitedAt: string;
}

export interface ActiveSession {
  sessionId: number;
  startedAt: string;
  lastActivity: string;
  pagesVisited: number;
}

export const analyticsApi = {
  startSession: async (): Promise<SessionResponse> => {
    const response = await client.post<ApiResponse<SessionResponse>>('/api/analytics/session/start');
    return response.data.data;
  },

  heartbeat: async (sessionId: number, pagePath?: string): Promise<SessionResponse> => {
    const response = await client.post<ApiResponse<SessionResponse>>('/api/analytics/session/heartbeat', {
      sessionId,
      pagePath,
    });
    return response.data.data;
  },

  endSession: async (sessionId: number): Promise<SessionResponse> => {
    const response = await client.post<ApiResponse<SessionResponse>>('/api/analytics/session/end', {
      sessionId,
    });
    return response.data.data;
  },

  logPageView: async (sessionId: number, pagePath: string, pageTitle?: string): Promise<PageViewResponse> => {
    const response = await client.post<ApiResponse<PageViewResponse>>('/api/analytics/pageview', {
      sessionId,
      pagePath,
      pageTitle,
    });
    return response.data.data;
  },

  getActiveSession: async (): Promise<ActiveSession | null> => {
    const response = await client.get<ApiResponse<ActiveSession | null>>('/api/analytics/session/active');
    return response.data.data;
  },
};

export default client;
