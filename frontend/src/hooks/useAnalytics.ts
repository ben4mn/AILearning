import { useEffect, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { analyticsApi } from '@/api/client';

const HEARTBEAT_INTERVAL = 30000; // 30 seconds
const SESSION_KEY = 'analytics_session_id';

/**
 * Analytics hook that tracks user sessions and page views
 * Should be used in a component that's rendered when user is authenticated
 */
export function useAnalytics(isAuthenticated: boolean) {
  const location = useLocation();
  const sessionIdRef = useRef<number | null>(null);
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastPageRef = useRef<string | null>(null);

  // Start session
  const startSession = useCallback(async () => {
    try {
      // Check for existing active session first
      const activeSession = await analyticsApi.getActiveSession();
      if (activeSession) {
        sessionIdRef.current = activeSession.sessionId;
        sessionStorage.setItem(SESSION_KEY, String(activeSession.sessionId));
      } else {
        const result = await analyticsApi.startSession();
        sessionIdRef.current = result.sessionId;
        sessionStorage.setItem(SESSION_KEY, String(result.sessionId));
      }
    } catch (error) {
      console.error('Failed to start analytics session:', error);
    }
  }, []);

  // End session
  const endSession = useCallback(async () => {
    const sessionId = sessionIdRef.current;
    if (sessionId) {
      try {
        await analyticsApi.endSession(sessionId);
      } catch (error) {
        console.error('Failed to end analytics session:', error);
      }
      sessionIdRef.current = null;
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, []);

  // Send heartbeat
  const sendHeartbeat = useCallback(async () => {
    const sessionId = sessionIdRef.current;
    if (!sessionId) return;

    try {
      const result = await analyticsApi.heartbeat(sessionId);
      // If a new session was created (old one expired), update our reference
      if (result.sessionId !== sessionId) {
        sessionIdRef.current = result.sessionId;
        sessionStorage.setItem(SESSION_KEY, String(result.sessionId));
      }
    } catch (error) {
      console.error('Failed to send heartbeat:', error);
    }
  }, []);

  // Log page view
  const logPageView = useCallback(async (path: string, title?: string) => {
    const sessionId = sessionIdRef.current;
    if (!sessionId) return;

    try {
      await analyticsApi.logPageView(sessionId, path, title);
    } catch (error) {
      console.error('Failed to log page view:', error);
    }
  }, []);

  // Start heartbeat interval
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    heartbeatIntervalRef.current = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
  }, [sendHeartbeat]);

  // Stop heartbeat interval
  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // Handle auth state changes
  useEffect(() => {
    if (isAuthenticated) {
      startSession().then(() => {
        startHeartbeat();
      });
    } else {
      stopHeartbeat();
      endSession();
    }

    return () => {
      stopHeartbeat();
    };
  }, [isAuthenticated, startSession, startHeartbeat, stopHeartbeat, endSession]);

  // Handle page navigation
  useEffect(() => {
    if (!isAuthenticated || !sessionIdRef.current) return;

    const currentPath = location.pathname;
    if (currentPath !== lastPageRef.current) {
      lastPageRef.current = currentPath;
      logPageView(currentPath, document.title);
    }
  }, [location.pathname, isAuthenticated, logPageView]);

  // Handle page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      const sessionId = sessionIdRef.current;
      if (sessionId) {
        // Use sendBeacon for reliable delivery on page unload
        const token = localStorage.getItem('token');
        if (token) {
          navigator.sendBeacon(
            '/api/analytics/session/end',
            new Blob([JSON.stringify({ sessionId })], { type: 'application/json' })
          );
        }
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  return {
    sessionId: sessionIdRef.current,
    endSession,
  };
}
