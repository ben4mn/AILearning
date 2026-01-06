import { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { MarkdownRenderer } from './MarkdownRenderer';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';
import type { LessonContent } from '@/types';

interface LessonViewerProps {
  lessonContent: LessonContent;
  topicSlug: string;
  hasQuiz?: boolean;
}

export function LessonViewer({ lessonContent, topicSlug, hasQuiz }: LessonViewerProps) {
  const navigate = useNavigate();
  const { lesson, content, previousLesson, nextLesson } = lessonContent;
  const { isLessonCompleted, markLessonComplete } = useProgressStore();
  const { isAuthenticated } = useAuthStore();

  const [isCompleted, setIsCompleted] = useState(false);
  const [isMarking, setIsMarking] = useState(false);
  const startTimeRef = useRef<number>(Date.now());

  // Check if already completed
  useEffect(() => {
    setIsCompleted(isLessonCompleted(lesson.id));
    startTimeRef.current = Date.now();
  }, [lesson.id, isLessonCompleted]);

  const handleMarkComplete = async () => {
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    setIsMarking(true);
    const timeSpent = Math.round((Date.now() - startTimeRef.current) / 1000);

    try {
      await markLessonComplete(lesson.id, timeSpent);
      setIsCompleted(true);
    } catch (error) {
      console.error('Failed to mark lesson complete:', error);
    } finally {
      setIsMarking(false);
    }
  };

  const handleNext = () => {
    if (nextLesson) {
      navigate(`/learn/${topicSlug}/${nextLesson.id}`);
    } else if (hasQuiz) {
      navigate(`/learn/${topicSlug}/quiz`);
    } else {
      navigate(`/learn/${topicSlug}`);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      {/* Breadcrumb */}
      <nav className="mb-6 text-sm">
        <ol className="flex items-center gap-2 text-gray-500">
          <li>
            <Link to="/learn" className="hover:text-primary-600">
              Learn
            </Link>
          </li>
          <li>/</li>
          <li>
            <Link to={`/learn/${topicSlug}`} className="hover:text-primary-600">
              {lesson.topicTitle}
            </Link>
          </li>
          <li>/</li>
          <li className="text-gray-900 font-medium">{lesson.title}</li>
        </ol>
      </nav>

      {/* Content */}
      <article className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <MarkdownRenderer content={content} />
      </article>

      {/* Completion & Navigation */}
      <div className="mt-8 flex flex-col sm:flex-row items-center justify-between gap-4 p-4 bg-gray-50 rounded-lg">
        {/* Mark Complete Button */}
        <div className="flex items-center gap-3">
          {isCompleted ? (
            <div className="flex items-center gap-2 text-green-600">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              <span className="font-medium">Completed</span>
            </div>
          ) : (
            <button
              onClick={handleMarkComplete}
              disabled={isMarking}
              className="btn-primary flex items-center gap-2"
            >
              {isMarking ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Mark Complete</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Navigation */}
        <div className="flex items-center gap-3">
          {previousLesson && (
            <Link
              to={`/learn/${topicSlug}/${previousLesson.id}`}
              className="flex items-center gap-1 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span>Previous</span>
            </Link>
          )}

          <button
            onClick={handleNext}
            className="flex items-center gap-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <span>
              {nextLesson ? 'Next Lesson' : hasQuiz ? 'Take Quiz' : 'Back to Topic'}
            </span>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
