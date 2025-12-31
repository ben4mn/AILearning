import { Link } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

export function Header() {
  const { user, isAuthenticated, logout } = useAuthStore();

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-secondary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">AI</span>
            </div>
            <span className="text-xl font-bold text-gray-900">AI History</span>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center space-x-6">
            {isAuthenticated ? (
              <>
                <Link to="/learn" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">
                  Learn
                </Link>
                <div className="flex items-center space-x-4">
                  <span className="text-gray-600">
                    {user?.displayName || user?.email}
                  </span>
                  <button
                    onClick={logout}
                    className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
                  >
                    Sign Out
                  </button>
                </div>
              </>
            ) : (
              <>
                <Link to="/login" className="text-gray-600 hover:text-gray-900 font-medium transition-colors">
                  Sign In
                </Link>
                <Link to="/register" className="btn-primary text-sm py-2">
                  Get Started
                </Link>
              </>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
