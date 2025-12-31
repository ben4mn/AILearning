import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import { Header } from '@/components/layout/Header';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { Home } from '@/pages/Home';
import { Login } from '@/pages/Login';
import { Register } from '@/pages/Register';
import { Learn } from '@/pages/Learn';
import { useAuthStore } from '@/stores/authStore';

function App() {
  const { fetchUser, token } = useAuthStore();

  useEffect(() => {
    // Try to restore session on app load
    if (token) {
      fetchUser();
    }
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/learn"
            element={
              <ProtectedRoute>
                <Learn />
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
