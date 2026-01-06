import { useEffect } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useCurriculumStore } from '@/stores/curriculumStore';
import { useProgressStore } from '@/stores/progressStore';
import { ViewToggle, LinearView, MindMap } from '@/components/navigation';

export function Learn() {
  const { user, isAuthenticated } = useAuthStore();
  const { viewMode } = useCurriculumStore();
  const { stats, fetchStats } = useProgressStore();

  useEffect(() => {
    if (isAuthenticated) {
      fetchStats();
    }
  }, [isAuthenticated, fetchStats]);

  const overallProgress = stats
    ? Math.round((stats.lessonsCompleted / Math.max(stats.totalLessons, 1)) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-1">
              {isAuthenticated ? `Welcome back, ${user?.displayName || 'Learner'}!` : 'AI History Curriculum'}
            </h1>
            <p className="text-gray-600">
              Explore the evolution of artificial intelligence from the 1940s to today.
            </p>
          </div>
          <ViewToggle />
        </div>

        {/* Progress Stats (for authenticated users) */}
        {isAuthenticated && stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{stats.lessonsCompleted}</p>
                  <p className="text-xs text-gray-500">Lessons Done</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{overallProgress}%</p>
                  <p className="text-xs text-gray-500">Progress</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{stats.quizzesPassed}</p>
                  <p className="text-xs text-gray-500">Quizzes Passed</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalTimeMinutes}</p>
                  <p className="text-xs text-gray-500">Minutes Spent</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Content View */}
        {viewMode === 'linear' ? <LinearView /> : <MindMap />}
      </div>
    </div>
  );
}
