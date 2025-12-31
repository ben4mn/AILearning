import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi } from '@/api/client';
import type { User, LoginCredentials, RegisterCredentials } from '@/types';

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  isAuthenticated: boolean;

  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (credentials: RegisterCredentials) => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      error: null,
      isAuthenticated: false,

      login: async (credentials: LoginCredentials) => {
        set({ isLoading: true, error: null });
        try {
          const { user, token } = await authApi.login(credentials);
          localStorage.setItem('token', token);
          set({ user, token, isAuthenticated: true, isLoading: false });
        } catch (error) {
          const message = error instanceof Error
            ? (error as { response?: { data?: { error?: string } } }).response?.data?.error || error.message
            : 'Login failed';
          set({ error: message, isLoading: false });
          throw error;
        }
      },

      register: async (credentials: RegisterCredentials) => {
        set({ isLoading: true, error: null });
        try {
          const { user, token } = await authApi.register(credentials);
          localStorage.setItem('token', token);
          set({ user, token, isAuthenticated: true, isLoading: false });
        } catch (error) {
          const message = error instanceof Error
            ? (error as { response?: { data?: { error?: string } } }).response?.data?.error || error.message
            : 'Registration failed';
          set({ error: message, isLoading: false });
          throw error;
        }
      },

      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, token: null, isAuthenticated: false, error: null });
      },

      fetchUser: async () => {
        const token = get().token || localStorage.getItem('token');
        if (!token) {
          set({ isAuthenticated: false });
          return;
        }

        set({ isLoading: true });
        try {
          const user = await authApi.getMe();
          set({ user, isAuthenticated: true, isLoading: false });
        } catch {
          localStorage.removeItem('token');
          set({ user: null, token: null, isAuthenticated: false, isLoading: false });
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
);
