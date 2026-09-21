# Guide de Déploiement - RAG System

Guide complet pour déployer le système RAG en local, staging et production sur AWS.

---

## Table des Matières

1. [Prérequis](#prérequis)
2. [Déploiement Local](#déploiement-local)
3. [Configuration AWS](#configuration-aws)
4. [Pipeline CI/CD](#pipeline-cicd)
5. [Déploiement en Production](#déploiement-en-production)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Prérequis

### Outils Requis

- **Docker** 20.10+
- **Docker Compose** 2.0+
- **AWS CLI** 2.0+
- **Python** 3.13
- **uv** 0.4.0
- **Git** 2.0+

### Comptes AWS

- Compte AWS avec accès IAM
- Credentials configurés (`aws configure`)
- Bedrock activé dans la région (eu-west-3)

### GitHub

- Repository GitHub
- GitHub Actions activé
- Secrets configurés

---

## Déploiement Local

### 1. Cloner le Repository

```bash
git clone https://github.com/your-org/rag-system.git
cd rag-system
```

### 2. Configurer l'Environnement

```bash
cp .env.example .env
```

Éditer `.env` avec vos valeurs:

```bash
AWS_BEARER_TOKEN_BEDROCK=your-token
AWS_REGION=eu-west-3
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Déployer avec Docker Compose

```bash
bash scripts/deploy-local.sh
```

Ou manuellement:

```bash
docker-compose up -d
```

### 4. Vérifier le Déploiement

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelle est la configuration réseau?"}'
```

Accéder à l'UI: http://localhost:8501

---

## Configuration AWS

### 1. Setup Infrastructure de Base

```bash
export AWS_REGION=eu-west-3
export ENVIRONMENT=dev

bash scripts/setup-aws-infrastructure.sh
```

Ce script crée:
- ECR repositories
- S3 buckets (logs de déploiement)
- CloudWatch Log Groups
- IAM roles pour GitHub Actions
- Secrets Manager

### 2. Configurer GitHub OIDC

Le script crée le provider OIDC. Mettre à jour la trust policy:

```bash
aws iam get-role --role-name GitHubActionsRole
```

Éditer le trust policy pour ajouter votre org/repo:

```json
{
  "StringLike": {
    "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:*"
  }
}
```

### 3. Pousser les Images vers ECR

```bash
export AWS_ACCOUNT_ID=123456789012
export IMAGE_TAG=$(git rev-parse --short HEAD)

bash scripts/push-to-ecr.sh
```

### 4. Configurer les Secrets GitHub

Dans GitHub: **Settings → Secrets → Actions**

Ajouter:

```
AWS_ACCOUNT_ID=123456789012
AWS_REGION=eu-west-3
DOMAIN_NAME=yourdomain.com
DEV_EC2_INSTANCE_ID=i-xxxxx
STAGING_EC2_INSTANCE_ID=i-xxxxx
PROD_EC2_INSTANCE_ID=i-xxxxx
BEDROCK_TOKEN=your-bedrock-token
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
CODECOV_TOKEN=your-codecov-token
```

---

## Pipeline CI/CD

### Architecture

```
Developer Push → GitHub → CI (Build & Test) → ECR
                             ↓
                        Auto Deploy Dev
                             ↓
                        Auto Deploy Staging
                             ↓
                   Manual Approval → Deploy Prod
```

### Workflows

#### 1. **CI - Build & Test** (`.github/workflows/ci.yml`)

Déclenché sur:
- Push sur `develop` ou `main`
- Pull Request

Steps:
1. Lint (Ruff)
2. Tests unitaires (pytest)
3. Security scan (Trivy)
4. Build Docker images
5. Push vers ECR

#### 2. **CD Dev** (`.github/workflows/cd-dev.yml`)

Déclenché sur:
- Push sur `develop`

Steps:
1. Deploy vers ECS Dev
2. Update EC2 instance
3. Smoke tests
4. Notification Slack

#### 3. **CD Staging** (`.github/workflows/cd-staging.yml`)

Déclenché sur:
- Push sur `main`

Steps:
1. Deploy vers ECS Staging
2. Update EC2 instance
3. Smoke tests + Integration tests
4. Notification Slack

#### 4. **CD Production** (`.github/workflows/cd-prod.yml`)

Déclenché sur:
- Manual trigger uniquement

Steps:
1. **Manual approval required**
2. Validate image exists
3. Blue/Green deployment
4. Health checks
5. Smoke tests production
6. Auto-rollback on failure
7. Notification Slack

---

## Déploiement en Production

### Processus de Déploiement

1. **Merge vers main** → Auto-deploy vers Staging

2. **Tester staging**
   ```bash
   curl https://rag-staging.yourdomain.com/health
   ```

3. **Déclencher Production Deploy**

   Dans GitHub Actions:
   - Aller dans **Actions → CD - Deploy to Production**
   - Cliquer **Run workflow**
   - Entrer l'image tag (ex: `abc1234` ou `latest`)
   - **Wait for manual approval**

4. **Approuver le déploiement**

   Un reviewer doit approuver dans l'onglet **Environments → production**

5. **Monitoring**

   Le workflow:
   - Valide l'image
   - Déploie en Blue/Green
   - Vérifie la santé des cibles
   - Roule back automatiquement si échec

### Rollback Manuel

Si besoin de rollback après déploiement:

```bash
aws ecs describe-services \
  --cluster rag-prod-cluster \
  --services streamlit-ui-service-prod \
  --query 'services[0].deployments' \
  --output json

PREVIOUS_TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition streamlit-ui-prod:N \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

aws ecs update-service \
  --cluster rag-prod-cluster \
  --service streamlit-ui-service-prod \
  --task-definition $PREVIOUS_TASK_DEF
```

---

## Monitoring

### CloudWatch Dashboards

Créer dashboard pour surveiller:

**Métriques ECS:**
- CPU/Memory utilization
- Task count
- Service deployment state

**Métriques ALB:**
- Request count
- Response times (p50, p95, p99)
- Error rates (4xx, 5xx)

**Métriques Custom:**
- Questions par minute
- Latence retrieval
- Latence génération
- Cache hit rate

### Logs

Consulter logs:

```bash
aws logs tail /ecs/rag-pipeline --follow

aws logs tail /ecs/streamlit-ui --follow
```

### Alarms

Configurer alarmes CloudWatch:

```bash
HIGH_ERROR_RATE: Target5xxCount > 10 for 5 min
HIGH_LATENCY: TargetResponseTime > 5s for 5 min
SERVICE_UNHEALTHY: HealthyHostCount < 2 for 2 min
```

---

## Troubleshooting

### Build Failed

**Symptôme:** CI workflow échoue au build

**Solutions:**
```bash
docker build -f Dockerfile.rag-pipeline -t test .

uv run ruff check .
uv run ruff format --check .
uv run pytest
```

### Deploy Failed - Image Not Found

**Symptôme:** ECR image not found

**Solutions:**
```bash
aws ecr describe-images --repository-name rag-pipeline

aws ecr list-images --repository-name rag-pipeline
```

### Service Unhealthy

**Symptôme:** ECS tasks failing health checks

**Solutions:**
```bash
aws ecs describe-services \
  --cluster rag-prod-cluster \
  --services streamlit-ui-service-prod

aws logs tail /ecs/streamlit-ui --since 10m
```

Vérifier:
- Environment variables correctes
- Secrets Manager accessible
- Bedrock endpoint accessible

### Blue/Green Deployment Stuck

**Symptôme:** Deployment reste en cours

**Solutions:**
```bash
aws ecs describe-services \
  --cluster rag-prod-cluster \
  --services streamlit-ui-service-prod \
  --query 'services[0].events[:10]'

aws ecs wait services-stable \
  --cluster rag-prod-cluster \
  --services streamlit-ui-service-prod
```

Force rollback si nécessaire (voir section Rollback).

### High Latency

**Symptôme:** Réponses lentes

**Solutions:**

1. Vérifier CloudWatch metrics
2. Augmenter resources ECS:
   ```yaml
   cpu: 2048
   memory: 4096
   ```
3. Scale out:
   ```bash
   aws ecs update-service \
     --cluster rag-prod-cluster \
     --service streamlit-ui-service-prod \
     --desired-count 3
   ```

---

## Commandes Utiles

### Docker Local

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down

docker-compose ps
docker-compose exec rag-pipeline bash
```

### AWS ECS

```bash
aws ecs list-clusters
aws ecs list-services --cluster rag-prod-cluster
aws ecs describe-services --cluster rag-prod-cluster --services <service-name>

aws ecs list-tasks --cluster rag-prod-cluster
aws ecs describe-tasks --cluster rag-prod-cluster --tasks <task-arn>
```

### ECR

```bash
aws ecr describe-repositories
aws ecr list-images --repository-name rag-pipeline
aws ecr describe-images --repository-name rag-pipeline --image-ids imageTag=latest
```

### Logs

```bash
aws logs tail /ecs/rag-pipeline --follow --since 10m

aws logs filter-log-events \
  --log-group-name /ecs/rag-pipeline \
  --filter-pattern "ERROR"
```

---

## Checklist Déploiement Production

Avant de déployer en prod, vérifier:

- [ ] Tests passent en CI
- [ ] Staging déployé et testé
- [ ] Image tag validé dans ECR
- [ ] Secrets à jour dans Secrets Manager
- [ ] Backup de la config actuelle
- [ ] Approvers notifiés
- [ ] Fenêtre de maintenance communiquée
- [ ] Plan de rollback prêt
- [ ] Monitoring dashboard ouvert

Après déploiement:

- [ ] Health checks passent
- [ ] Smoke tests OK
- [ ] Métriques dans les SLO
- [ ] Pas d'erreurs dans logs
- [ ] Équipe notifiée du succès

---

## Support

Pour assistance:
- Consulter logs CloudWatch
- Vérifier GitHub Actions runs
- Contacter l'équipe DevOps
- Escalader via Slack #rag-system-alerts
