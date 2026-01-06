import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useCurriculumStore, ERA_INFO } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { useAuthStore } from '@/stores/authStore';
import { quizApi } from '@/api/client';
import { QuizPlayer } from '@/components/quiz';
import type { QuizWithQuestions, QuizResult } from '@/types';

export function Quiz() {
  const { topicSlug } = useParams<{ topicSlug: string }>();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();
  const { currentTopic, fetchTopic } = useCurriculumStore();
  const { fetchOverview, fetchStats } = useProgressStore();
  const [quiz, setQuiz] = useState<QuizWithQuestions | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch topic if not loaded
  useEffect(() => {
    if (topicSlug && (!currentTopic || currentTopic.slug !== topicSlug)) {
      fetchTopic(topicSlug);
    }
  }, [topicSlug, currentTopic, fetchTopic]);

  // Fetch quiz
  useEffect(() => {
    const loadQuiz = async () => {
      if (!topicSlug) return;

      setIsLoading(true);
      setError(null);

      try {
        const quizData = await quizApi.getByTopicSlug(topicSlug);
        setQuiz(quizData);
      } catch (err) {
        console.error('Failed to load quiz:', err);
        setError('Failed to load quiz. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    loadQuiz();
  }, [topicSlug]);

  const handleSubmit = async (answers: Record<number, string>): Promise<QuizResult> => {
    if (!quiz) {
      throw new Error('No quiz loaded');
    }

    return await quizApi.submit(quiz.id, { answers });
  };

  const handleComplete = async (result: QuizResult) => {
    // Refresh progress data
    await Promise.all([fetchOverview(), fetchStats()]);

    if (result.passed) {
      // Navigate to next topic or back to learn page
      // For now, just go back to the topic page
      navigate(`/learn/${topicSlug}`);
    }
    // If not passed, QuizPlayer shows retry option
  };

  const eraInfo = currentTopic ? ERA_INFO[currentTopic.era as keyof typeof ERA_INFO] : null;

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg border border-gray-200 p-8 max-w-md text-center">
          <div className="w-16 h-16 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
              />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">Sign In Required</h2>
          <p className="text-gray-600 mb-6">
            You need to be signed in to take quizzes and track your progress.
          </p>
          <div className="flex flex-col gap-3">
            <Link to="/login" className="btn-primary">
              Sign In
            </Link>
            <Link to="/register" className="text-primary-600 hover:text-primary-700">
              Create an account
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3 text-gray-600">
          <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <span>Loading quiz...</span>
        </div>
      </div>
    );
  }

  if (error || !quiz) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Quiz Not Available</h2>
          <p className="text-gray-600 mb-4">{error || 'This topic does not have a quiz yet.'}</p>
          <Link to={topicSlug ? `/learn/${topicSlug}` : '/learn'} className="btn-primary">
            Back to Topic
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Breadcrumb */}
          <nav className="flex items-center gap-2 text-sm text-gray-500 mb-4">
            <Link to="/learn" className="hover:text-gray-700">
              Learn
            </Link>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <Link to={`/learn/${topicSlug}`} className="hover:text-gray-700">
              {currentTopic?.title || 'Topic'}
            </Link>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="text-gray-900">Quiz</span>
          </nav>

          <div className="flex items-center gap-4">
            <div className="flex-shrink-0 w-12 h-12 bg-amber-500 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
                />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                {eraInfo && (
                  <span
                    className="text-xs font-medium px-2 py-0.5 rounded"
                    style={{
                      backgroundColor: `${eraInfo.color}20`,
                      color: eraInfo.color,
                    }}
                  >
                    {eraInfo.name}
                  </span>
                )}
              </div>
              <h1 className="text-2xl font-bold text-gray-900">{quiz.title}</h1>
              <p className="text-gray-600">
                {quiz.questions.length} questions • {quiz.passingScore}% to pass
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quiz Content */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <QuizPlayer quiz={quiz} onSubmit={handleSubmit} onComplete={handleComplete} />
      </div>
    </div>
  );
}
