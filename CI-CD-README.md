# 🚀 CI/CD Pipeline - Quick Reference

Pipeline automatisé GitHub Actions → Docker → AWS pour le système RAG.

---

## 📋 Quick Start

### Déploiement Local (5 min)

```bash
make deploy-local
```

Accès:
- UI: http://localhost:8501
- API: http://localhost:8080

### Setup AWS (1ère fois - 2h)

```bash
# 1. Setup infrastructure
bash scripts/setup-aws-infrastructure.sh

# 2. Configure GitHub Secrets (voir docs/CI-CD-SETUP.md)

# 3. Push images
export AWS_ACCOUNT_ID=123456789012
make push-ecr

# 4. Premier deploy
git push origin develop  # Auto-deploy Dev
```

---

## 📁 Documentation Complète

| Document | Contenu |
|----------|---------|
| **[docs/CI-CD-SETUP.md](docs/CI-CD-SETUP.md)** | ⚡ Setup rapide en 6 étapes (2h) |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | 📚 Guide complet de déploiement |
| **[docs/CI-CD-FILES.md](docs/CI-CD-FILES.md)** | 📑 Index de tous les fichiers |

---

## 🔄 Workflows GitHub Actions

| Workflow | Trigger | Durée | Description |
|----------|---------|-------|-------------|
| `ci.yml` | Push, PR | 5-8 min | Tests + Build + ECR |
| `cd-dev.yml` | Push develop | 3-5 min | Deploy Dev |
| `cd-staging.yml` | Push main | 5-7 min | Deploy Staging |
| `cd-prod.yml` | Manual | 8-12 min | Deploy Prod (approval required) |

---

## 🛠️ Commandes Utiles

### Docker Local

```bash
make docker-build    # Build images
make docker-up       # Start services
make docker-down     # Stop services
make logs            # View logs
make health          # Check health
```

### AWS

```bash
make push-ecr        # Push images to ECR
make setup-aws       # Setup infrastructure
bash scripts/validate-deployment.sh prod  # Pre-deploy checks
```

### Tests

```bash
make test            # Run tests
make lint            # Lint code
make validate        # Full validation
```

---

## 🐳 Docker Images

### Build Local

```bash
docker build -f Dockerfile.rag-pipeline -t rag-pipeline:latest .
docker build -f Dockerfile.streamlit -t streamlit-ui:latest .
```

### Run Local

```bash
docker-compose up -d
```

---

## 📦 Structure Fichiers

```
.github/workflows/     # GitHub Actions workflows
docs/                  # Documentation
scripts/               # Scripts de déploiement
src/api/               # FastAPI server
Dockerfile.*           # Docker images
docker-compose*.yml    # Compose configs
Makefile               # Commands
```

---

## 🔐 Secrets Requis

### GitHub Secrets

```
AWS_ACCOUNT_ID, DOMAIN_NAME, BEDROCK_TOKEN,
DEV/STAGING/PROD_EC2_INSTANCE_ID, SLACK_WEBHOOK_URL
```

### AWS Secrets Manager

```
rag-system/{env}/config
  → AWS_BEARER_TOKEN_BEDROCK, ANTHROPIC_API_KEY
```

---

## 🚦 Workflow de Déploiement

### Feature Development

```bash
git checkout -b feature/my-feature
# Code...
git push origin feature/my-feature
# Create PR → develop
# Merge → Auto-deploy Dev
```

### Release to Staging

```bash
git checkout main
git merge develop
git push origin main
# Auto-deploy Staging
```

### Release to Production

```
GitHub → Actions → CD - Deploy to Production
→ Run workflow → Wait approval → Deploy
```

---

## 📊 Monitoring

```bash
# Logs CloudWatch
aws logs tail /ecs/rag-pipeline --follow

# ECS Services
aws ecs describe-services --cluster rag-prod-cluster --services streamlit-ui-service-prod

# Health
curl https://rag.yourdomain.com/health
```

---

## 🆘 Troubleshooting

### Build failed

```bash
make validate         # Check lint + tests
docker-compose build  # Test local build
```

### Deploy failed

```bash
bash scripts/validate-deployment.sh prod  # Pre-checks
aws ecs describe-services ...             # Check ECS
aws logs tail /ecs/streamlit-ui --since 10m  # Check logs
```

### Rollback

```bash
# Via GitHub Actions: Re-run previous successful workflow
# Ou manual:
aws ecs update-service --cluster rag-prod-cluster \
  --service streamlit-ui-service-prod \
  --task-definition streamlit-ui-prod:PREVIOUS_REVISION
```

---

## 💰 Coûts

| Service | Coût/mois |
|---------|-----------|
| ECR | ~$2 |
| ECS | ~$30 |
| EC2 | ~$30 |
| ALB | ~$20 |
| CloudWatch | ~$10 |
| **Total** | **~$92** |

---

## ✅ Checklist Pre-Production

- [ ] Tests pass (`make test`)
- [ ] Lint clean (`make lint`)
- [ ] Staging tested
- [ ] Images in ECR
- [ ] Secrets configured
- [ ] Approval ready
- [ ] Monitoring dashboard open

---

## 📖 Documentation Complète

Pour plus de détails, consulter:

1. **[docs/CI-CD-SETUP.md](docs/CI-CD-SETUP.md)** - Setup initial détaillé
2. **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Guide complet
3. **[docs/CI-CD-FILES.md](docs/CI-CD-FILES.md)** - Index fichiers

---

## 🎯 Next Steps

1. Setup AWS infrastructure: `bash scripts/setup-aws-infrastructure.sh`
2. Configure GitHub Secrets
3. Push images: `make push-ecr`
4. Test local: `make deploy-local`
5. Deploy dev: `git push origin develop`
6. Deploy prod: GitHub Actions manual trigger
