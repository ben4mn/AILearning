import { Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

export function Home() {
  const { isAuthenticated } = useAuthStore();

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Discover the{' '}
            <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              History of AI
            </span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-10">
            From the Turing Test to modern neural networks, explore the fascinating evolution of artificial intelligence through interactive lessons, mind maps, and quizzes.
          </p>
          <div className="flex justify-center gap-4">
            {isAuthenticated ? (
              <Link to="/learn" className="btn-primary text-lg py-4 px-8">
                Continue Learning
              </Link>
            ) : (
              <>
                <Link to="/register" className="btn-primary text-lg py-4 px-8">
                  Start Learning Free
                </Link>
                <Link to="/login" className="btn-secondary text-lg py-4 px-8">
                  Sign In
                </Link>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
          Learn Your Way
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="card text-center">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Mind Map View</h3>
            <p className="text-gray-600">
              Explore topics visually with an interactive mind map showing how concepts connect and evolve.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-16 h-16 bg-secondary-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-secondary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Chapter View</h3>
            <p className="text-gray-600">
              Prefer a linear approach? Follow a structured curriculum from foundations to cutting-edge AI.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-3">Quiz Gates</h3>
            <p className="text-gray-600">
              Test your knowledge with quizzes that unlock new topics as you master each concept.
            </p>
          </div>
        </div>
      </section>

      {/* Eras Preview */}
      <section className="bg-gray-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">
            Journey Through Time
          </h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { era: '1940s-1960s', title: 'Foundations', desc: 'Turing, early computing, birth of AI' },
              { era: '1970s-1980s', title: 'Expert Systems', desc: 'Rule-based AI, knowledge engineering' },
              { era: '1990s-2000s', title: 'Machine Learning', desc: 'Statistical methods, SVMs, early neural nets' },
              { era: '2010s-Now', title: 'Deep Learning', desc: 'Transformers, LLMs, modern AI' },
            ].map((item) => (
              <div key={item.era} className="bg-gray-800 rounded-xl p-6 hover:bg-gray-700 transition-colors">
                <p className="text-primary-400 text-sm font-medium mb-2">{item.era}</p>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-gray-400 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-600">
          <p>Built with curiosity and code</p>
        </div>
      </footer>
    </div>
  );
}
