import { create } from 'zustand';
import { progressApi } from '@/api/client';
import type { ProgressStats, ProgressOverviewItem, TopicProgress } from '@/types';

interface ProgressState {
  // Data
  stats: ProgressStats | null;
  overview: ProgressOverviewItem[];
  currentTopicProgress: TopicProgress | null;
  completedLessons: Set<number>;
  passedQuizzes: Set<number>;

  // UI State
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchStats: () => Promise<void>;
  fetchOverview: () => Promise<void>;
  fetchTopicProgress: (topicSlug: string) => Promise<void>;
  fetchCompletedItems: () => Promise<void>;
  markLessonComplete: (lessonId: number, timeSpentSeconds?: number) => Promise<void>;
  isLessonCompleted: (lessonId: number) => boolean;
  isQuizPassed: (quizId: number) => boolean;
  clearError: () => void;
}

export const useProgressStore = create<ProgressState>()((set, get) => ({
  stats: null,
  overview: [],
  currentTopicProgress: null,
  completedLessons: new Set(),
  passedQuizzes: new Set(),
  isLoading: false,
  error: null,

  fetchStats: async () => {
    set({ isLoading: true, error: null });
    try {
      const stats = await progressApi.getStats();
      set({ stats, isLoading: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch progress';
      set({ error: message, isLoading: false });
    }
  },

  fetchOverview: async () => {
    set({ isLoading: true, error: null });
    try {
      const overview = await progressApi.getOverview();
      set({ overview, isLoading: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch progress overview';
      set({ error: message, isLoading: false });
    }
  },

  fetchTopicProgress: async (topicSlug: string) => {
    set({ isLoading: true, error: null });
    try {
      const progress = await progressApi.getByTopic(topicSlug);
      set({ currentTopicProgress: progress, isLoading: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch topic progress';
      set({ error: message, isLoading: false, currentTopicProgress: null });
    }
  },

  fetchCompletedItems: async () => {
    try {
      const { completedLessons, passedQuizzes } = await progressApi.getCompleted();
      set({
        completedLessons: new Set(completedLessons),
        passedQuizzes: new Set(passedQuizzes),
      });
    } catch (error) {
      console.error('Failed to fetch completed items:', error);
    }
  },

  markLessonComplete: async (lessonId: number, timeSpentSeconds?: number) => {
    try {
      await progressApi.completeLesson(lessonId, timeSpentSeconds);
      // Update local state
      const newCompleted = new Set(get().completedLessons);
      newCompleted.add(lessonId);
      set({ completedLessons: newCompleted });
      // Refresh stats
      get().fetchStats();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to mark lesson complete';
      set({ error: message });
      throw error;
    }
  },

  isLessonCompleted: (lessonId: number) => {
    return get().completedLessons.has(lessonId);
  },

  isQuizPassed: (quizId: number) => {
    return get().passedQuizzes.has(quizId);
  },

  clearError: () => {
    set({ error: null });
  },
}));
