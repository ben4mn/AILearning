import { useState } from 'react';
import { QuestionCard } from './QuestionCard';
import type { QuizWithQuestions, QuizResult } from '@/types';

interface QuizPlayerProps {
  quiz: QuizWithQuestions;
  onSubmit: (answers: Record<number, string>) => Promise<QuizResult>;
  onComplete: (result: QuizResult) => void;
}

export function QuizPlayer({ quiz, onSubmit, onComplete }: QuizPlayerProps) {
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<QuizResult | null>(null);

  const questions = quiz.questions;
  const currentQuestion = questions[currentQuestionIndex];
  const answeredCount = Object.keys(answers).length;
  const allAnswered = answeredCount === questions.length;

  const handleSelectAnswer = (questionId: number, answer: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: answer }));
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const quizResult = await onSubmit(answers);
      setResult(quizResult);
    } catch (error) {
      console.error('Failed to submit quiz:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleContinue = () => {
    if (result) {
      onComplete(result);
    }
  };

  // Results view
  if (result) {
    return (
      <div className="space-y-6">
        {/* Result Summary */}
        <div
          className={`p-6 rounded-lg text-center ${
            result.passed ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}
        >
          <div
            className={`text-5xl font-bold mb-2 ${
              result.passed ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {result.score}%
          </div>
          <div
            className={`text-lg font-medium ${
              result.passed ? 'text-green-800' : 'text-red-800'
            }`}
          >
            {result.passed ? 'Congratulations! You passed!' : 'Not quite. Try again!'}
          </div>
          <div className="text-sm text-gray-600 mt-2">
            {result.correctCount} of {result.totalQuestions} correct (need {quiz.passingScore}% to
            pass)
          </div>
        </div>

        {/* Review Questions */}
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-900">Review Your Answers</h3>
          {questions.map((question, index) => {
            const feedback = result.feedback.find((f) => f.questionId === question.id);
            return (
              <QuestionCard
                key={question.id}
                questionNumber={index + 1}
                totalQuestions={questions.length}
                questionText={question.questionText}
                options={question.options}
                selectedAnswer={answers[question.id]}
                onSelectAnswer={() => {}}
                disabled
                showResult
                correctAnswer={feedback?.correctAnswer}
                explanation={feedback?.explanation}
              />
            );
          })}
        </div>

        {/* Continue Button */}
        <div className="flex justify-center">
          <button onClick={handleContinue} className="btn-primary px-8">
            {result.passed ? 'Continue' : 'Try Again'}
          </button>
        </div>
      </div>
    );
  }

  // Quiz view
  return (
    <div className="space-y-6">
      {/* Progress */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          {answeredCount} of {questions.length} answered
        </div>
        <div className="flex gap-1">
          {questions.map((q, index) => (
            <button
              key={q.id}
              onClick={() => setCurrentQuestionIndex(index)}
              className={`w-8 h-8 rounded-full text-xs font-medium transition-colors ${
                answers[q.id]
                  ? 'bg-primary-500 text-white'
                  : index === currentQuestionIndex
                  ? 'bg-primary-100 text-primary-700 border-2 border-primary-500'
                  : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              }`}
            >
              {index + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Current Question */}
      <QuestionCard
        questionNumber={currentQuestionIndex + 1}
        totalQuestions={questions.length}
        questionText={currentQuestion.questionText}
        options={currentQuestion.options}
        selectedAnswer={answers[currentQuestion.id]}
        onSelectAnswer={(answer) => handleSelectAnswer(currentQuestion.id, answer)}
      />

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={handlePrevious}
          disabled={currentQuestionIndex === 0}
          className="flex items-center gap-1 px-4 py-2 text-gray-600 hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span>Previous</span>
        </button>

        <div className="flex gap-3">
          {currentQuestionIndex < questions.length - 1 ? (
            <button onClick={handleNext} className="btn-primary flex items-center gap-1">
              <span>Next</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!allAnswered || isSubmitting}
              className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Submitting...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  <span>Submit Quiz</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {!allAnswered && currentQuestionIndex === questions.length - 1 && (
        <p className="text-sm text-amber-600 text-center">
          Please answer all questions before submitting.
        </p>
      )}
    </div>
  );
}
