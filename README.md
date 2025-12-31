# AI History Learning Platform

An interactive educational web application designed to take learners on a comprehensive journey through the history and evolution of Artificial Intelligence—from its theoretical foundations in the 1940s to modern concepts like transformers, tokenization, vectorization, and Retrieval-Augmented Generation (RAG).

## Vision

Learning AI shouldn't be a linear slog through dense textbooks. This platform reimagines AI education as an **explorable landscape** where learners can:

- **Follow the main timeline** of AI development like chapters in a book
- **Branch off into fascinating tangents** when curiosity strikes—exploring the philosophy of mind, the mathematics of neural networks, or the stories of pioneering researchers
- **Always find their way back** to the main narrative without losing progress

The goal is to create an experience where getting "sidetracked" isn't a distraction—it's the point. Real understanding comes from seeing how ideas connect, influence each other, and build toward the AI systems we use today.

## Key Features

### Dual Navigation System
- **Mind Map View**: A visual graph where topics appear as interconnected nodes. Click any node to explore. See how "The Turing Test" connects to "Philosophy of Mind" and "Early Computing." Follow your curiosity.
- **Linear Chapter View**: A traditional book-style progression for those who prefer structured learning. Work through eras chronologically, completing lessons in sequence.

Switch between views anytime. Your progress syncs across both.

### Era-Based Curriculum
The content is organized into historical eras, each capturing a distinct phase of AI development:

| Era | Period | Key Topics |
|-----|--------|------------|
| **Foundations** | 1940s-1960s | Turing Test, early computing, symbolic AI, perceptrons |
| **AI Winter & Expert Systems** | 1970s-1980s | Knowledge-based systems, LISP, expert systems, limitations |
| **Machine Learning Renaissance** | 1990s-2000s | Statistical methods, SVMs, decision trees, early neural nets |
| **Deep Learning Revolution** | 2010s-Present | CNNs, RNNs, transformers, attention mechanisms |
| **Modern AI** | 2020s | Large language models, tokenization, embeddings, RAG, agents |

### Gamified Learning
- **Progress Tracking**: Visual progress bars show completion across topics and eras
- **Quiz Gates**: Short quizzes between sections reinforce learning and unlock new content
- **Achievements**: Earn badges for milestones (complete an era, perfect quiz scores, learning streaks)
- **Encouragement System**: Contextual messages celebrate progress and motivate continued learning

### User Accounts
- Create an account to save progress across devices
- Track time spent learning
- Review quiz history and performance
- Resume exactly where you left off

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19, Vite, TypeScript |
| **Styling** | Tailwind CSS v4 |
| **State Management** | Zustand |
| **Graph Visualization** | React Flow |
| **Backend** | Node.js, Express |
| **Database** | PostgreSQL |
| **Authentication** | JWT (JSON Web Tokens) |
| **Containerization** | Docker, Docker Compose |

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ben4mn/AILearning.git
   cd AILearning
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your own secrets
   ```

3. **Start the application**
   ```bash
   docker compose up --build
   ```

4. **Access the app**
   - Frontend: http://localhost:3051
   - Backend API: http://localhost:3050
   - Database: localhost:5433

### Development Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Stop all services
docker compose down

# Reset database (removes all data)
docker compose down -v && docker compose up --build
```

## Project Structure

```
ai-history-app/
├── frontend/                 # React application
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   │   ├── auth/         # Login, Register, ProtectedRoute
│   │   │   ├── layout/       # Header, Sidebar, ProgressBar
│   │   │   ├── navigation/   # MindMapView, LinearView
│   │   │   ├── content/      # LessonCard, LessonContent
│   │   │   ├── quiz/         # QuizGate, QuizQuestion
│   │   │   └── gamification/ # ProgressRing, Achievements
│   │   ├── pages/            # Route-level components
│   │   ├── stores/           # Zustand state stores
│   │   ├── hooks/            # Custom React hooks
│   │   ├── api/              # API client
│   │   └── types/            # TypeScript definitions
│   └── Dockerfile
│
├── backend/                  # Express API server
│   └── src/
│       ├── routes/           # API route handlers
│       ├── services/         # Business logic
│       ├── middleware/       # Auth, validation, errors
│       ├── config/           # Database, environment
│       └── db/               # Schema, migrations
│
├── content/                  # Lesson content (Markdown)
│   ├── curriculum.json       # Course structure definition
│   └── lessons/              # Markdown files by era/topic
│
├── docker-compose.yml        # Container orchestration
└── CLAUDE.md                 # Development guidelines
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Create new account |
| `/api/auth/login` | POST | Authenticate user |
| `/api/auth/me` | GET | Get current user |
| `/api/topics` | GET | List all topics (for mind map) |
| `/api/topics/linear` | GET | Topics in chapter order |
| `/api/lessons/:id` | GET | Get lesson content |
| `/api/progress` | GET | User's learning progress |
| `/api/progress/lesson/:id` | POST | Mark lesson complete |
| `/api/quiz/:topicSlug` | GET | Get quiz for topic |
| `/api/quiz/:id/submit` | POST | Submit quiz answers |

## Roadmap

### Phase 1: Foundation ✅
- [x] Project structure and Docker setup
- [x] PostgreSQL database with full schema
- [x] User authentication (register, login, JWT)
- [x] Basic frontend with protected routes

### Phase 2: Content System (Next)
- [ ] Define curriculum.json structure
- [ ] Create sample lessons (3-5 topics)
- [ ] Content API endpoints
- [ ] Linear chapter navigation
- [ ] Markdown lesson rendering

### Phase 3: Mind Map
- [ ] React Flow integration
- [ ] Interactive topic graph
- [ ] View toggle (map ↔ chapters)
- [ ] Position sync between views

### Phase 4: Progress & Gamification
- [ ] Progress tracking API
- [ ] Visual progress indicators
- [ ] Achievement system
- [ ] Learning streaks

### Phase 5: Quizzes
- [ ] Quiz API endpoints
- [ ] Quiz UI components
- [ ] Gate logic (require passing to proceed)
- [ ] Results and review

### Phase 6: Polish
- [ ] Animations and transitions
- [ ] Dark mode
- [ ] Mobile responsiveness
- [ ] Error handling improvements

## Contributing

This project is in active development. Contributions, suggestions, and feedback are welcome.

## License

MIT License - See LICENSE file for details.

---

*Built with curiosity about AI's past and excitement for its future.*
