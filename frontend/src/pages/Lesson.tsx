import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useCurriculumStore, ERA_INFO } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';
import { LessonViewer } from '@/components/content';

export function Lesson() {
  const { topicSlug, lessonId } = useParams<{ topicSlug: string; lessonId: string }>();
  const { isAuthenticated } = useAuthStore();
  const { currentTopic, currentLesson, isLoading, error, fetchTopic, fetchLesson } = useCurriculumStore();
  const { fetchOverview } = useProgressStore();

  // Fetch topic if not loaded or different
  useEffect(() => {
    if (topicSlug && (!currentTopic || currentTopic.slug !== topicSlug)) {
      fetchTopic(topicSlug);
    }
  }, [topicSlug, currentTopic, fetchTopic]);

  // Fetch lesson content
  useEffect(() => {
    if (lessonId) {
      fetchLesson(parseInt(lessonId, 10));
    }
  }, [lessonId, fetchLesson]);

  // Fetch progress for authenticated users
  useEffect(() => {
    if (isAuthenticated) {
      fetchOverview();
    }
  }, [isAuthenticated, fetchOverview]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-600">
          <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <span>Loading lesson...</span>
        </div>
      </div>
    );
  }

  if (error || !currentLesson) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Lesson Not Found</h2>
          <p className="text-gray-600 mb-4">{error || 'The requested lesson could not be found.'}</p>
          <Link to={topicSlug ? `/learn/${topicSlug}` : '/learn'} className="btn-primary">
            Back to Topic
          </Link>
        </div>
      </div>
    );
  }

  const eraInfo = currentTopic ? ERA_INFO[currentTopic.era as keyof typeof ERA_INFO] : null;
  const totalLessons = currentTopic?.lessons?.length || 1;
  const currentIndex = currentTopic?.lessons?.findIndex((l) => l.id === currentLesson.lesson.id) ?? 0;
  const hasQuiz = !!currentTopic?.quiz;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between">
            {/* Topic info */}
            <div className="flex items-center gap-3">
              {eraInfo && (
                <span
                  className="hidden sm:inline-block text-xs font-medium px-2 py-0.5 rounded"
                  style={{
                    backgroundColor: `${eraInfo.color}20`,
                    color: eraInfo.color,
                  }}
                >
                  {eraInfo.name}
                </span>
              )}
              <span className="text-sm text-gray-600">{currentTopic?.title}</span>
            </div>

            {/* Progress indicator */}
            <span className="text-sm text-gray-500">
              Lesson {currentIndex + 1} of {totalLessons}
            </span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="py-8">
        <LessonViewer
          lessonContent={currentLesson}
          topicSlug={topicSlug || ''}
          hasQuiz={hasQuiz}
        />
      </div>
    </div>
  );
}
