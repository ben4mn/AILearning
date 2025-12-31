# CLAUDE.md - AI History Learning Platform

## Project Overview
An interactive educational web app for learning AI history, featuring dual navigation (mind map + linear chapters), user accounts with progress tracking, and gamified learning with quiz gates.

## Tech Stack
- **Frontend**: React 19 + Vite + TypeScript + Tailwind CSS v4
- **State Management**: Zustand
- **Graph Visualization**: React Flow (mind map)
- **Backend**: Node.js + Express
- **Database**: PostgreSQL
- **Auth**: JWT (email/password)
- **Container**: Docker + docker-compose

## Ports
- Backend API: `3050`
- Frontend Dev: `3051`
- PostgreSQL: `5433` (external) / `5432` (internal)

## Development Commands

### Docker (recommended)
```bash
# Start all services
docker-compose up --build

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop all services
docker-compose down

# Reset database
docker-compose down -v && docker-compose up --build
```

### Local Development (without Docker)
```bash
# Backend
cd backend
npm install
npm run dev

# Frontend
cd frontend
npm install
npm run dev
```

## Environment Variables

Copy `.env.example` to `.env` and configure:
```bash
DB_PASSWORD=your_secure_password
JWT_SECRET=your_jwt_secret_key
```

## API Endpoints

### Auth
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login, returns JWT
- `GET /api/auth/me` - Get current user (requires auth)

### Content
- `GET /api/topics` - Get all topics
- `GET /api/topics/:slug` - Get single topic with lessons
- `GET /api/topics/linear` - Get topics in chapter order
- `GET /api/lessons/:id` - Get lesson content

### Progress
- `GET /api/progress` - Get user's progress
- `POST /api/progress/lesson/:id` - Mark lesson complete

### Quiz
- `GET /api/quiz/:topicSlug` - Get quiz for topic
- `POST /api/quiz/:id/submit` - Submit quiz attempt

## Code Style

### Backend (JavaScript)
- Use ES6+ features
- Async/await for asynchronous operations
- Consistent error handling with try/catch
- Use middleware for auth and validation

### Frontend (TypeScript)
- Functional components with hooks
- Zustand for global state
- Type all props and state
- Use absolute imports from `@/`

## Testing
```bash
# Backend tests
cd backend && npm test

# Frontend tests
cd frontend && npm test
```

## Database
Schema is auto-initialized via Docker. To manually run:
```bash
psql -h localhost -p 5433 -U aihistory -d aihistory -f backend/src/db/schema.sql
```

## File Organization
- `/backend/src/routes/` - API route handlers
- `/backend/src/services/` - Business logic
- `/backend/src/middleware/` - Express middleware
- `/frontend/src/components/` - React components
- `/frontend/src/pages/` - Page components
- `/frontend/src/stores/` - Zustand stores
- `/content/` - Lesson content files (Markdown)
