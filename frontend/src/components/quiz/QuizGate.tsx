import { Link } from 'react-router-dom';

interface QuizGateProps {
  topicSlug: string;
  quizTitle: string;
  passingScore: number;
  questionCount: number;
  isPassed?: boolean;
  bestScore?: number | null;
}

export function QuizGate({
  topicSlug,
  quizTitle,
  passingScore,
  questionCount,
  isPassed = false,
  bestScore,
}: QuizGateProps) {
  if (isPassed) {
    return (
      <div className="p-6 bg-green-50 border border-green-200 rounded-lg">
        <div className="flex items-center gap-3 mb-3">
          <div className="flex-shrink-0 w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-green-800">{quizTitle}</h3>
            <p className="text-sm text-green-700">
              Passed with {bestScore}%
            </p>
          </div>
        </div>
        <Link
          to={`/learn/${topicSlug}/quiz`}
          className="inline-flex items-center text-sm text-green-700 hover:text-green-800"
        >
          <span>Review your answers</span>
          <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>
    );
  }

  return (
    <div className="p-6 bg-amber-50 border border-amber-200 rounded-lg">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 bg-amber-500 rounded-full flex items-center justify-center">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-amber-800 mb-1">{quizTitle}</h3>
          <p className="text-sm text-amber-700 mb-3">
            Complete this quiz to unlock the next topic. You'll need {passingScore}% to pass.
          </p>
          <div className="flex items-center gap-4 text-sm text-amber-700 mb-4">
            <span>{questionCount} questions</span>
            <span>Need {passingScore}% to pass</span>
            {bestScore !== null && bestScore !== undefined && (
              <span>Best score: {bestScore}%</span>
            )}
          </div>
          <Link
            to={`/learn/${topicSlug}/quiz`}
            className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>Start Quiz</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
