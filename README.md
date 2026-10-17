# dubhacks_2026

Phone in the SO-101 gripper, laptop calling the shots. Perceptron Mk1.5 parses the request and picks the angle. The arm does a cinematic move. The phone gets back a clip of exactly the length you asked for.

DubHacks 2026, Perceptron Robotics track.

| Doc | What it is |
|---|---|
| [PLAN.md](PLAN.md) | Schedule, tracks, risks |
| [CONTRACTS.md](CONTRACTS.md) | Frozen message and function shapes |
| [AGENTS.md](AGENTS.md) | Who owns which folder, how to start, team log |
| [WORKSHOP.md](WORKSHOP.md) | Isaac 0.5 API, Mk1.5, arm contract, event deadlines |

```
web/       PWA (Vite + React)                 Frontend
server/    FastAPI orchestrator               Backend
robot/     arm control + VLA client           Robotics
ml/        Mk1.5 prompts + VLA train          ML / Video
shared/    schemas, examples, constants
```

Day-1 setup is in [AGENTS.md](AGENTS.md). Copy `.env.example` to `.env` and keep the real keys there.
