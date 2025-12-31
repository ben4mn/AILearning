import { Navigate } from 'react-router-dom';
import { LoginForm } from '@/components/auth/LoginForm';
import { useAuthStore } from '@/stores/authStore';

export function Login() {
  const { isAuthenticated } = useAuthStore();

  // Redirect if already logged in
  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <LoginForm />
    </div>
  );
}
