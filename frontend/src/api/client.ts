import axios, { AxiosError } from 'axios';
import type { ApiResponse, ApiError, AuthResponse, LoginCredentials, RegisterCredentials, User } from '@/types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3050';

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle response errors
client.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    if (error.response?.status === 401) {
      // Clear token on unauthorized
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  register: async (credentials: RegisterCredentials): Promise<AuthResponse> => {
    const response = await client.post<ApiResponse<AuthResponse>>('/api/auth/register', credentials);
    return response.data.data;
  },

  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await client.post<ApiResponse<AuthResponse>>('/api/auth/login', credentials);
    return response.data.data;
  },

  getMe: async (): Promise<User> => {
    const response = await client.get<ApiResponse<{ user: User }>>('/api/auth/me');
    return response.data.data.user;
  },
};

export default client;
