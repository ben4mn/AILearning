import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import { Header } from '@/components/layout/Header';
import { Home, Login, Register, Learn, Topic, Lesson, Quiz } from '@/pages';
import { useAuthStore } from '@/stores/authStore';
import { useAnalytics } from '@/hooks/useAnalytics';

function AppContent() {
  const { fetchUser, token, isAuthenticated } = useAuthStore();

  // Initialize analytics tracking
  useAnalytics(isAuthenticated);

  useEffect(() => {
    // Try to restore session on app load
    if (token) {
      fetchUser();
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/learn" element={<Learn />} />
        <Route path="/learn/:topicSlug" element={<Topic />} />
        <Route path="/learn/:topicSlug/:lessonId" element={<Lesson />} />
        <Route path="/learn/:topicSlug/quiz" element={<Quiz />} />
      </Routes>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
