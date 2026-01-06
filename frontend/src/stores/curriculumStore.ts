import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { topicsApi, lessonsApi } from '@/api/client';
import type {
  TopicListItem,
  TopicWithLessons,
  TopicConnection,
  LessonContent,
  ViewMode,
} from '@/types';

interface CurriculumState {
  // Data
  topics: TopicListItem[];
  connections: TopicConnection[];
  currentTopic: TopicWithLessons | null;
  currentLesson: LessonContent | null;
  topicsByEra: Record<string, TopicListItem[]>;

  // UI State
  viewMode: ViewMode;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchTopics: () => Promise<void>;
  fetchTopicsByEra: () => Promise<void>;
  fetchConnections: () => Promise<void>;
  fetchTopic: (slug: string) => Promise<void>;
  fetchLesson: (lessonId: number) => Promise<void>;
  setViewMode: (mode: ViewMode) => void;
  clearCurrentTopic: () => void;
  clearCurrentLesson: () => void;
  clearError: () => void;
}

// Era metadata for display
export const ERA_INFO: Record<string, { name: string; color: string; order: number }> = {
  foundations: { name: 'Foundations (1940s-1960s)', color: '#3B82F6', order: 1 },
  'ai-winter': { name: 'AI Winter & Expert Systems (1970s-1980s)', color: '#8B5CF6', order: 2 },
  'ml-renaissance': { name: 'ML Renaissance (1990s-2000s)', color: '#10B981', order: 3 },
  'deep-learning': { name: 'Deep Learning Revolution (2010s)', color: '#F59E0B', order: 4 },
  'modern-ai': { name: 'Modern AI (2020s)', color: '#EF4444', order: 5 },
};

export const useCurriculumStore = create<CurriculumState>()(
  persist(
    (set) => ({
      topics: [],
      connections: [],
      currentTopic: null,
      currentLesson: null,
      topicsByEra: {},
      viewMode: 'linear',
      isLoading: false,
      error: null,

      fetchTopics: async () => {
        set({ isLoading: true, error: null });
        try {
          const topics = await topicsApi.getLinear();
          set({ topics, isLoading: false });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to fetch topics';
          set({ error: message, isLoading: false });
        }
      },

      fetchTopicsByEra: async () => {
        set({ isLoading: true, error: null });
        try {
          const topicsByEra = await topicsApi.getByEra();
          set({ topicsByEra, isLoading: false });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to fetch topics';
          set({ error: message, isLoading: false });
        }
      },

      fetchConnections: async () => {
        try {
          const connections = await topicsApi.getConnections();
          set({ connections });
        } catch (error) {
          console.error('Failed to fetch connections:', error);
        }
      },

      fetchTopic: async (slug: string) => {
        set({ isLoading: true, error: null });
        try {
          const topic = await topicsApi.getBySlug(slug);
          set({ currentTopic: topic, isLoading: false });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to fetch topic';
          set({ error: message, isLoading: false, currentTopic: null });
        }
      },

      fetchLesson: async (lessonId: number) => {
        set({ isLoading: true, error: null });
        try {
          const lesson = await lessonsApi.getContent(lessonId);
          set({ currentLesson: lesson, isLoading: false });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Failed to fetch lesson';
          set({ error: message, isLoading: false, currentLesson: null });
        }
      },

      setViewMode: (mode: ViewMode) => {
        set({ viewMode: mode });
      },

      clearCurrentTopic: () => {
        set({ currentTopic: null });
      },

      clearCurrentLesson: () => {
        set({ currentLesson: null });
      },

      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'curriculum-storage',
      partialize: (state) => ({ viewMode: state.viewMode }),
    }
  )
);
