# CI/CD Pipeline - Index des Fichiers

Index complet de tous les fichiers créés pour le pipeline CI/CD.

---

## 📦 Docker & Containers

### Dockerfiles

| Fichier | Description | Taille estimée |
|---------|-------------|----------------|
| `Dockerfile.rag-pipeline` | Image RAG Pipeline (API FastAPI) | ~500 MB |
| `Dockerfile.streamlit` | Image Streamlit UI | ~450 MB |
| `.dockerignore` | Exclusions Docker build | - |

### Docker Compose

| Fichier | Usage |
|---------|-------|
| `docker-compose.yml` | Dev local |
| `docker-compose.prod.yml` | Production AWS (avec EFS, ALB) |

---

## 🔄 GitHub Actions Workflows

### Workflows CI/CD

| Fichier | Déclenché par | Durée | Description |
|---------|---------------|-------|-------------|
| `.github/workflows/ci.yml` | Push, PR | ~5-8 min | Tests, lint, security scan, build images |
| `.github/workflows/cd-dev.yml` | Push develop | ~3-5 min | Deploy automatique vers Dev |
| `.github/workflows/cd-staging.yml` | Push main | ~5-7 min | Deploy automatique vers Staging |
| `.github/workflows/cd-prod.yml` | Manual | ~8-12 min | Deploy production avec approval |
| `.github/workflows/scheduled-tests.yml` | Cron 2AM daily | ~10-15 min | Tests nightly + benchmarks |

### Workflow Features

**ci.yml:**
- ✅ Python 3.13 + uv
- ✅ Ruff lint + format check
- ✅ Pytest avec coverage
- ✅ Trivy security scan
- ✅ Build multi-stage Docker images
- ✅ Push vers ECR avec tags (commit SHA + latest)

**cd-dev.yml:**
- ✅ Deploy ECS task
- ✅ Update EC2 Ollama instance via SSM
- ✅ Smoke tests
- ✅ Slack notifications

**cd-staging.yml:**
- ✅ Deploy ECS
- ✅ Integration tests
- ✅ Performance checks

**cd-prod.yml:**
- ✅ Manual approval required
- ✅ Blue/Green deployment
- ✅ Health checks avec seuils
- ✅ Auto-rollback on failure
- ✅ Deployment audit logs → S3

---

## 🛠️ Scripts de Déploiement

### Scripts Shell

| Script | Usage | Durée |
|--------|-------|-------|
| `scripts/deploy-local.sh` | Déploiement local complet | ~5 min |
| `scripts/push-to-ecr.sh` | Build + Push images vers ECR | ~10 min |
| `scripts/setup-aws-infrastructure.sh` | Setup initial AWS (ECR, IAM, S3, etc.) | ~15 min |
| `scripts/validate-deployment.sh` | Pre-deployment checks | ~2 min |

**deploy-local.sh:**
```bash
bash scripts/deploy-local.sh

# Fait:
# 1. Check requirements (Docker, .env)
# 2. Build images
# 3. Start services
# 4. Wait for health
# 5. Show status + URLs
```

**push-to-ecr.sh:**
```bash
export AWS_ACCOUNT_ID=123456789012
export IMAGE_TAG=v1.0.0

bash scripts/push-to-ecr.sh

# Fait:
# 1. Auth ECR
# 2. Create repos if needed
# 3. Build images
# 4. Tag + push (commit SHA + latest)
```

**setup-aws-infrastructure.sh:**
```bash
export ENVIRONMENT=prod
bash scripts/setup-aws-infrastructure.sh

# Crée:
# - ECR repositories
# - S3 buckets (deployment logs)
# - CloudWatch Log Groups
# - IAM role GitHubActionsRole
# - Secrets Manager
# - GitHub OIDC provider
```

**validate-deployment.sh:**
```bash
bash scripts/validate-deployment.sh prod

# Vérifie:
# - AWS credentials
# - Docker running
# - Images in ECR
# - Secrets Manager
# - ECS cluster health
# - Tests pass
# - Lint clean
```

---

## 📄 Configuration Files

### Environment

| Fichier | Description |
|---------|-------------|
| `.env.example` | Template variables d'environnement |
| `.env` | Variables locales (gitignored) |

**Variables clés:**
```bash
# AWS
AWS_REGION=eu-west-3
AWS_BEARER_TOKEN_BEDROCK=xxx
AWS_ACCOUNT_ID=123456789012

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Docker/ECR
ECR_REGISTRY=123456789012.dkr.ecr.eu-west-3.amazonaws.com
IMAGE_TAG=latest

# Environment
ENVIRONMENT=prod
LOG_LEVEL=INFO
```

### Makefile

| Commande | Description |
|----------|-------------|
| `make help` | Liste toutes les commandes |
| `make install` | Install deps (uv sync) |
| `make test` | Run tests |
| `make lint` | Lint code |
| `make docker-build` | Build images |
| `make docker-up` | Start services |
| `make deploy-local` | Full local deploy |
| `make push-ecr` | Push to ECR |
| `make setup-aws` | Setup AWS infra |
| `make logs` | View logs |
| `make health` | Check health |
| `make clean` | Clean artifacts |

---

## 📚 Documentation

| Fichier | Contenu |
|---------|---------|
| `docs/DEPLOYMENT.md` | Guide complet de déploiement (20+ pages) |
| `docs/CI-CD-SETUP.md` | Quick start CI/CD (2h setup) |
| `docs/CI-CD-FILES.md` | Ce fichier - index des fichiers |

**DEPLOYMENT.md couvre:**
- Prérequis
- Déploiement local
- Configuration AWS complète
- Pipeline CI/CD détaillé
- Déploiement production
- Monitoring & alerting
- Troubleshooting
- Commandes utiles

**CI-CD-SETUP.md couvre:**
- Setup rapide en 6 étapes
- Configuration GitHub Secrets
- Branch protection rules
- Premier déploiement
- Workflow quotidien
- Troubleshooting rapide

---

## 🐍 Code Python

### API Server

| Fichier | Description |
|---------|-------------|
| `src/api/__init__.py` | Package init |
| `src/api/server.py` | FastAPI server avec endpoints /health et /api/ask |

**Endpoints:**
- `GET /health` → Health check
- `POST /api/ask` → Question answering

---

## 📊 Structure Complète

```
context-aware-qa/
├── .github/
│   └── workflows/
│       ├── ci.yml                      # CI: Build & Test
│       ├── cd-dev.yml                  # CD: Deploy Dev
│       ├── cd-staging.yml              # CD: Deploy Staging
│       ├── cd-prod.yml                 # CD: Deploy Prod (manual)
│       └── scheduled-tests.yml         # Nightly tests
│
├── docs/
│   ├── DEPLOYMENT.md                   # Guide complet
│   ├── CI-CD-SETUP.md                  # Quick start
│   └── CI-CD-FILES.md                  # This file
│
├── scripts/
│   ├── deploy-local.sh                 # Deploy local
│   ├── push-to-ecr.sh                  # Push to ECR
│   ├── setup-aws-infrastructure.sh     # Setup AWS
│   └── validate-deployment.sh          # Pre-deploy checks
│
├── src/
│   └── api/
│       ├── __init__.py
│       └── server.py                   # FastAPI server
│
├── Dockerfile.rag-pipeline             # RAG API image
├── Dockerfile.streamlit                # UI image
├── .dockerignore                       # Docker exclusions
├── docker-compose.yml                  # Dev local
├── docker-compose.prod.yml             # Production AWS
├── .env.example                        # Template env vars
├── Makefile                            # Commands
└── README.md                           # Main docs (updated)
```

---

## 🚀 Quick Commands

### Déploiement Local

```bash
make deploy-local
```

### Build & Push ECR

```bash
export AWS_ACCOUNT_ID=123456789012
make push-ecr
```

### Validation Pre-Deploy

```bash
bash scripts/validate-deployment.sh prod
```

### Monitoring

```bash
# Logs
make logs

# Health
make health

# AWS ECS
aws ecs describe-services --cluster rag-prod-cluster --services streamlit-ui-service-prod
```

---

## 🔐 Secrets Requis

### GitHub Secrets

```
AWS_ACCOUNT_ID
AWS_REGION
DOMAIN_NAME
DEV_EC2_INSTANCE_ID
STAGING_EC2_INSTANCE_ID
PROD_EC2_INSTANCE_ID
BEDROCK_TOKEN
SLACK_WEBHOOK_URL
CODECOV_TOKEN (optional)
```

### AWS Secrets Manager

```
rag-system/dev/config
rag-system/staging/config
rag-system/prod/config

Contenu:
{
  "AWS_BEARER_TOKEN_BEDROCK": "xxx",
  "ANTHROPIC_API_KEY": "xxx"
}
```

---

## 📈 Métriques & Monitoring

### CloudWatch Logs

```
/ecs/rag-pipeline
/ecs/streamlit-ui
/ec2/ollama
```

### Métriques à surveiller

- Deployment frequency
- Lead time for changes
- Mean time to recovery (MTTR)
- Change failure rate
- ECS CPU/Memory
- ALB response times
- Error rates

---

## 💰 Coûts Estimés

| Service | Coût mensuel |
|---------|--------------|
| ECR (2 repos) | ~$2 |
| ECS Fargate | ~$30 |
| EC2 (Ollama) | ~$30 |
| ALB | ~$20 |
| CloudWatch | ~$10 |
| **Total** | **~$92/mois** |

---

## ✅ Checklist Déploiement

### Initial Setup

- [ ] AWS infrastructure créée (`setup-aws-infrastructure.sh`)
- [ ] GitHub OIDC trust policy configurée
- [ ] GitHub Secrets ajoutés
- [ ] GitHub Environment "production" créé avec approvers
- [ ] Branch protection rules configurées
- [ ] Images initiales dans ECR

### Chaque Déploiement

- [ ] Tests passent localement
- [ ] Lint clean
- [ ] PR reviewed et merged
- [ ] CI workflow vert
- [ ] Staging testé
- [ ] Backup config prod
- [ ] Approval obtenu (prod only)
- [ ] Monitoring dashboard ouvert

---

## 📞 Support

**Documentation:**
- Main: `README.md`
- Deployment: `docs/DEPLOYMENT.md`
- Quick start: `docs/CI-CD-SETUP.md`

**Commandes utiles:**
```bash
make help
bash scripts/validate-deployment.sh
```

**Troubleshooting:**
Voir `docs/DEPLOYMENT.md#troubleshooting`
