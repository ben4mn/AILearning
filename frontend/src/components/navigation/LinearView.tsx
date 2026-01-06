import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useCurriculumStore, ERA_INFO } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';
import type { TopicListItem } from '@/types';

interface TopicCardProps {
  topic: TopicListItem;
  progress?: {
    completedLessons: number;
    percentComplete: number;
    quizPassed: boolean;
  };
}

function TopicCard({ topic, progress }: TopicCardProps) {
  const eraInfo = ERA_INFO[topic.era || ''] || { color: '#6B7280', name: 'Unknown' };

  return (
    <Link
      to={`/learn/${topic.slug}`}
      className="block p-4 bg-white rounded-lg border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ backgroundColor: eraInfo.color }}
            />
            <span className="text-xs text-gray-500">{eraInfo.name}</span>
          </div>
          <h3 className="font-semibold text-gray-900 truncate">{topic.title}</h3>
          {topic.description && (
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{topic.description}</p>
          )}
          <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
            <span>{topic.lessonCount} lessons</span>
            <span>{topic.estimatedMinutes} min</span>
            {topic.hasQuiz && <span className="text-primary-600">Quiz</span>}
          </div>
        </div>

        {progress && (
          <div className="flex flex-col items-end gap-1">
            <div className="text-sm font-medium text-gray-900">
              {progress.percentComplete}%
            </div>
            <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 rounded-full transition-all"
                style={{ width: `${progress.percentComplete}%` }}
              />
            </div>
            {progress.quizPassed && (
              <span className="text-xs text-green-600 flex items-center gap-1">
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
                Passed
              </span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}

export function LinearView() {
  const { topics, isLoading, error, fetchTopics } = useCurriculumStore();
  const { overview, fetchOverview } = useProgressStore();
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    fetchTopics();
    if (isAuthenticated) {
      fetchOverview();
    }
  }, [fetchTopics, fetchOverview, isAuthenticated]);

  // Group topics by era
  const topicsByEra = topics.reduce((acc, topic) => {
    const era = topic.era || 'unknown';
    if (!acc[era]) acc[era] = [];
    acc[era].push(topic);
    return acc;
  }, {} as Record<string, TopicListItem[]>);

  // Create progress lookup
  const progressLookup = overview.reduce((acc, item) => {
    acc[item.slug] = {
      completedLessons: item.completedLessons,
      percentComplete: item.percentComplete,
      quizPassed: item.quizPassed,
    };
    return acc;
  }, {} as Record<string, { completedLessons: number; percentComplete: number; quizPassed: boolean }>);

  // Sort eras by order
  const sortedEras = Object.keys(topicsByEra).sort((a, b) => {
    return (ERA_INFO[a]?.order || 99) - (ERA_INFO[b]?.order || 99);
  });

  if (isLoading && topics.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">{error}</p>
        <button
          onClick={() => fetchTopics()}
          className="mt-4 text-primary-600 hover:text-primary-700"
        >
          Try again
        </button>
      </div>
    );
  }

  if (topics.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">No topics available yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {sortedEras.map((era) => {
        const eraInfo = ERA_INFO[era] || { name: era, color: '#6B7280' };
        const eraTopics = topicsByEra[era];

        return (
          <div key={era}>
            <div className="flex items-center gap-3 mb-4">
              <div
                className="w-1 h-6 rounded-full"
                style={{ backgroundColor: eraInfo.color }}
              />
              <h2 className="text-lg font-semibold text-gray-900">{eraInfo.name}</h2>
              <span className="text-sm text-gray-500">
                {eraTopics.length} {eraTopics.length === 1 ? 'topic' : 'topics'}
              </span>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {eraTopics.map((topic) => (
                <TopicCard
                  key={topic.id}
                  topic={topic}
                  progress={isAuthenticated ? progressLookup[topic.slug] : undefined}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
