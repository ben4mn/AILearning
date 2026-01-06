import { Link } from 'react-router-dom';
import type { Lesson } from '@/types';
import { useProgressStore } from '@/stores/progressStore';

interface LessonListProps {
  lessons: Lesson[];
  topicSlug: string;
  currentLessonId?: number;
}

export function LessonList({ lessons, topicSlug, currentLessonId }: LessonListProps) {
  const { isLessonCompleted } = useProgressStore();

  return (
    <div className="space-y-2">
      {lessons.map((lesson, index) => {
        const isCompleted = isLessonCompleted(lesson.id);
        const isCurrent = lesson.id === currentLessonId;

        return (
          <Link
            key={lesson.id}
            to={`/learn/${topicSlug}/${lesson.id}`}
            className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
              isCurrent
                ? 'bg-primary-50 border border-primary-200'
                : 'hover:bg-gray-50 border border-transparent'
            }`}
          >
            {/* Completion indicator */}
            <div
              className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                isCompleted
                  ? 'bg-green-500 text-white'
                  : isCurrent
                  ? 'bg-primary-500 text-white'
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              {isCompleted ? (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                index + 1
              )}
            </div>

            {/* Lesson info */}
            <div className="flex-1 min-w-0">
              <div
                className={`font-medium truncate ${
                  isCurrent ? 'text-primary-700' : 'text-gray-900'
                }`}
              >
                {lesson.title}
              </div>
              <div className="text-xs text-gray-500 capitalize">
                {lesson.lessonType}
              </div>
            </div>

            {/* Type indicator */}
            {lesson.lessonType === 'interactive' && (
              <span className="flex-shrink-0 text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                Interactive
              </span>
            )}
            {lesson.lessonType === 'video' && (
              <span className="flex-shrink-0 text-xs px-2 py-0.5 bg-red-100 text-red-700 rounded">
                Video
              </span>
            )}
          </Link>
        );
      })}
    </div>
  );
}
