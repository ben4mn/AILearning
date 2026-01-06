import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useCurriculumStore, ERA_INFO } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';
import { LessonList } from '@/components/content';
import { QuizGate } from '@/components/quiz';

export function Topic() {
  const { topicSlug } = useParams<{ topicSlug: string }>();
  const { isAuthenticated } = useAuthStore();
  const { currentTopic, isLoading, error, fetchTopic } = useCurriculumStore();
  const { currentTopicProgress, fetchTopicProgress } = useProgressStore();

  useEffect(() => {
    if (topicSlug) {
      fetchTopic(topicSlug);
    }
  }, [topicSlug, fetchTopic]);

  useEffect(() => {
    if (isAuthenticated && topicSlug) {
      fetchTopicProgress(topicSlug);
    }
  }, [isAuthenticated, topicSlug, fetchTopicProgress]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-600">
          <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <span>Loading topic...</span>
        </div>
      </div>
    );
  }

  if (error || !currentTopic) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Topic Not Found</h2>
          <p className="text-gray-600 mb-4">{error || 'The requested topic could not be found.'}</p>
          <Link to="/learn" className="btn-primary">
            Back to Learning
          </Link>
        </div>
      </div>
    );
  }

  const eraInfo = ERA_INFO[currentTopic.era as keyof typeof ERA_INFO];

  // Build completed lessons set from topic progress
  const completedLessonIds = new Set(
    currentTopicProgress?.lessons
      .filter((l) => l.completed)
      .map((l) => l.id) || []
  );
  const quizPassed = currentTopicProgress?.quiz?.passed || false;
  const bestScore = currentTopicProgress?.quiz?.bestScore;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-sm text-gray-500 mb-6">
          <Link to="/learn" className="hover:text-gray-700">
            Learn
          </Link>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span className="text-gray-900">{currentTopic.title}</span>
        </nav>

        {/* Topic Header */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
          <div className="flex items-start gap-4">
            {/* Era Badge */}
            <div
              className="flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center text-white text-xl"
              style={{ backgroundColor: eraInfo?.color || '#6B7280' }}
            >
              {currentTopic.linearOrder}
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="text-xs font-medium px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: `${eraInfo?.color}20`,
                    color: eraInfo?.color,
                  }}
                >
                  {eraInfo?.name || currentTopic.era}
                </span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900 mb-2">{currentTopic.title}</h1>
              <p className="text-gray-600">{currentTopic.description}</p>

              {/* Stats */}
              <div className="flex items-center gap-4 mt-4 text-sm text-gray-500">
                <div className="flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                    />
                  </svg>
                  <span>{currentTopic.lessons?.length || 0} lessons</span>
                </div>
                <div className="flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>{currentTopic.estimatedMinutes} min</span>
                </div>
                {isAuthenticated && currentTopicProgress && (
                  <div className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span>
                      {completedLessonIds.size} / {currentTopic.lessons?.length || 0} completed
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Lessons List */}
        {currentTopic.lessons && currentTopic.lessons.length > 0 && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Lessons</h2>
            <LessonList
              lessons={currentTopic.lessons}
              topicSlug={currentTopic.slug}
            />
          </div>
        )}

        {/* Quiz Gate */}
        {currentTopic.quiz && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Topic Quiz</h2>
            <QuizGate
              topicSlug={currentTopic.slug}
              quizTitle={currentTopic.quiz.title}
              passingScore={currentTopic.quiz.passingScore}
              questionCount={currentTopic.quiz.questionCount || 5}
              isPassed={quizPassed}
              bestScore={bestScore}
            />
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
          <Link
            to="/learn"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span>Back to all topics</span>
          </Link>

          {currentTopic.lessons && currentTopic.lessons.length > 0 && (
            <Link
              to={`/learn/${currentTopic.slug}/${currentTopic.lessons[0].id}`}
              className="btn-primary flex items-center gap-2"
            >
              <span>Start Learning</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
