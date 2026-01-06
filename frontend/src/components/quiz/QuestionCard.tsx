interface QuestionCardProps {
  questionNumber: number;
  totalQuestions: number;
  questionText: string;
  options: string[];
  selectedAnswer?: string;
  onSelectAnswer: (answer: string) => void;
  disabled?: boolean;
  // Review mode props
  showResult?: boolean;
  correctAnswer?: string;
  explanation?: string;
}

export function QuestionCard({
  questionNumber,
  totalQuestions,
  questionText,
  options,
  selectedAnswer,
  onSelectAnswer,
  disabled = false,
  showResult = false,
  correctAnswer,
  explanation,
}: QuestionCardProps) {
  const isCorrect = showResult && selectedAnswer === correctAnswer;
  const isIncorrect = showResult && selectedAnswer && selectedAnswer !== correctAnswer;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-medium text-gray-500">
          Question {questionNumber} of {totalQuestions}
        </span>
        {showResult && (
          <span
            className={`text-sm font-medium px-2 py-0.5 rounded ${
              isCorrect
                ? 'bg-green-100 text-green-700'
                : isIncorrect
                ? 'bg-red-100 text-red-700'
                : 'bg-gray-100 text-gray-700'
            }`}
          >
            {isCorrect ? 'Correct' : isIncorrect ? 'Incorrect' : 'Not answered'}
          </span>
        )}
      </div>

      {/* Question */}
      <h3 className="text-lg font-medium text-gray-900 mb-4">{questionText}</h3>

      {/* Options */}
      <div className="space-y-3">
        {options.map((option, index) => {
          const isSelected = selectedAnswer === option;
          const isThisCorrect = showResult && option === correctAnswer;
          const isThisWrong = showResult && isSelected && option !== correctAnswer;

          let optionClasses =
            'flex items-center gap-3 p-4 rounded-lg border-2 transition-colors cursor-pointer';

          if (showResult) {
            if (isThisCorrect) {
              optionClasses += ' border-green-500 bg-green-50';
            } else if (isThisWrong) {
              optionClasses += ' border-red-500 bg-red-50';
            } else {
              optionClasses += ' border-gray-200 bg-gray-50';
            }
          } else if (isSelected) {
            optionClasses += ' border-primary-500 bg-primary-50';
          } else {
            optionClasses += ' border-gray-200 hover:border-primary-300 hover:bg-gray-50';
          }

          if (disabled) {
            optionClasses += ' cursor-not-allowed';
          }

          return (
            <button
              key={index}
              type="button"
              onClick={() => !disabled && onSelectAnswer(option)}
              disabled={disabled}
              className={optionClasses}
            >
              {/* Radio indicator */}
              <div
                className={`flex-shrink-0 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  showResult
                    ? isThisCorrect
                      ? 'border-green-500 bg-green-500'
                      : isThisWrong
                      ? 'border-red-500 bg-red-500'
                      : 'border-gray-300'
                    : isSelected
                    ? 'border-primary-500 bg-primary-500'
                    : 'border-gray-300'
                }`}
              >
                {(isSelected || isThisCorrect) && (
                  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                )}
              </div>

              {/* Option text */}
              <span
                className={`text-left ${
                  showResult
                    ? isThisCorrect
                      ? 'text-green-800 font-medium'
                      : isThisWrong
                      ? 'text-red-800'
                      : 'text-gray-600'
                    : isSelected
                    ? 'text-primary-800 font-medium'
                    : 'text-gray-700'
                }`}
              >
                {option}
              </span>
            </button>
          );
        })}
      </div>

      {/* Explanation (in review mode) */}
      {showResult && explanation && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2">
            <svg
              className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-blue-800">{explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
}
