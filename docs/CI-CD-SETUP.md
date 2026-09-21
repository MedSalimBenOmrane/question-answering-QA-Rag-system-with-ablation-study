# CI/CD Pipeline - Quick Setup Guide

Guide rapide pour mettre en place le pipeline CI/CD GitHub Actions → AWS.

---

## Vue d'Ensemble

```
Git Push → GitHub Actions → Docker Build → ECR → Deploy (Dev/Staging/Prod)
```

**Temps de setup:** ~2 heures  
**Coût estimé:** ~$50-100/mois (AWS)

---

## Prérequis (5 min)

- [ ] Compte AWS avec droits administrateur
- [ ] Repository GitHub avec admin access
- [ ] AWS CLI configuré localement
- [ ] Docker installé

---

## Étape 1: Setup AWS Infrastructure (30 min)

### 1.1 Créer l'infrastructure

```bash
cd context-aware-qa

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=eu-west-3
export ENVIRONMENT=prod

bash scripts/setup-aws-infrastructure.sh
```

Ce script crée:
- ✅ ECR repositories (rag-pipeline, streamlit-ui)
- ✅ S3 buckets (deployment logs)
- ✅ CloudWatch Log Groups
- ✅ IAM role GitHubActionsRole
- ✅ Secrets Manager
- ✅ GitHub OIDC provider

### 1.2 Configurer le trust policy GitHub

Éditer la trust policy du role `GitHubActionsRole`:

```bash
aws iam get-role --role-name GitHubActionsRole --query 'Role.AssumeRolePolicyDocument' > trust-policy.json
```

Éditer `trust-policy.json` et remplacer:
```json
"StringLike": {
  "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:*"
}
```

Avec votre org/repo (ex: `"repo:anthropics/rag-system:*"`).

Appliquer:
```bash
aws iam update-assume-role-policy \
  --role-name GitHubActionsRole \
  --policy-document file://trust-policy.json
```

### 1.3 Configurer Secrets Manager

```bash
aws secretsmanager update-secret \
  --secret-id rag-system/prod/config \
  --secret-string '{
    "AWS_BEARER_TOKEN_BEDROCK": "YOUR_REAL_TOKEN",
    "ANTHROPIC_API_KEY": "YOUR_REAL_KEY"
  }'
```

---

## Étape 2: Configurer GitHub Secrets (10 min)

Dans GitHub: **Settings → Secrets and variables → Actions → New repository secret**

Ajouter ces secrets:

| Secret | Valeur | Description |
|--------|--------|-------------|
| `AWS_ACCOUNT_ID` | `123456789012` | AWS Account ID |
| `DOMAIN_NAME` | `yourdomain.com` | Nom de domaine |
| `DEV_EC2_INSTANCE_ID` | `i-xxxxx` | Instance EC2 Dev |
| `STAGING_EC2_INSTANCE_ID` | `i-xxxxx` | Instance EC2 Staging |
| `PROD_EC2_INSTANCE_ID` | `i-xxxxx` | Instance EC2 Prod |
| `BEDROCK_TOKEN` | `your-token` | Token Bedrock |
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/...` | Webhook Slack |
| `CODECOV_TOKEN` | `your-token` | Token Codecov (optionnel) |

---

## Étape 3: Configurer GitHub Environments (5 min)

### 3.1 Créer l'environment Production

1. **Settings → Environments → New environment**
2. Nom: `production`
3. Cocher **Required reviewers**
4. Ajouter les reviewers (personnes qui peuvent approuver les deploys prod)
5. **Save protection rules**

---

## Étape 4: Branching Strategy (5 min)

### 4.1 Créer les branches

```bash
git checkout -b develop
git push -u origin develop

git checkout main
```

### 4.2 Configurer les Branch Protection Rules

**Pour `main`:**

Settings → Branches → Add rule

- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require approvals: 2
- ✅ Require status checks to pass before merging
  - Add: `test`, `security-scan`, `build-images`
- ✅ Require signed commits
- ✅ Do not allow bypassing the above settings

**Pour `develop`:**

- Branch name pattern: `develop`
- ✅ Require a pull request before merging
- ✅ Require approvals: 1
- ✅ Require status checks: `test`

---

## Étape 5: Premier Déploiement (30 min)

### 5.1 Push initial vers develop

```bash
git checkout develop
git add .
git commit -m "Setup CI/CD pipeline"
git push origin develop
```

GitHub Actions va:
1. ✅ Run tests
2. ✅ Build images
3. ✅ Push vers ECR
4. ✅ Deploy vers Dev

### 5.2 Merge vers main (Staging)

```bash
git checkout main
git merge develop
git push origin main
```

GitHub Actions va:
1. ✅ Build & test
2. ✅ Deploy vers Staging
3. ✅ Run integration tests

### 5.3 Deploy vers Production

1. **GitHub → Actions → CD - Deploy to Production**
2. **Run workflow**
3. **Entrer image tag:** `latest`
4. **Wait for approval** (l'approver configuré reçoit une notification)
5. **Approver review** dans l'onglet **Environments**
6. Le workflow continue et déploie en production

---

## Étape 6: Vérification (10 min)

### 6.1 Vérifier les workflows

```bash
# Via GitHub UI
GitHub → Actions

# Ou via CLI
gh run list
gh run view <run-id>
```

### 6.2 Vérifier les images ECR

```bash
aws ecr describe-images --repository-name rag-pipeline --region eu-west-3
aws ecr describe-images --repository-name streamlit-ui --region eu-west-3
```

### 6.3 Tester les endpoints

```bash
# Dev
curl https://rag-dev.yourdomain.com/health

# Staging
curl https://rag-staging.yourdomain.com/health

# Prod (après déploiement)
curl https://rag.yourdomain.com/health
```

---

## Workflow Quotidien

### Développement de features

```bash
git checkout develop
git checkout -b feature/my-feature

# Coder...

git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature

# Créer PR sur GitHub: feature/my-feature → develop
# Après approbation et merge → auto-deploy Dev
```

### Release vers Staging

```bash
git checkout main
git merge develop
git push origin main

# Auto-deploy vers Staging
```

### Release vers Production

1. Tester staging
2. GitHub → Actions → CD - Deploy to Production
3. Run workflow avec tag
4. Attendre approval
5. Approve
6. Monitoring pendant le deploy

---

## Troubleshooting

### Workflow échoue: "role-to-assume not allowed"

**Cause:** Trust policy GitHub OIDC mal configurée

**Fix:**
```bash
aws iam get-role --role-name GitHubActionsRole

# Vérifier que trust policy contient:
# "StringLike": {
#   "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:*"
# }
```

### Images ne se push pas vers ECR

**Cause:** Credentials ECR expirés

**Fix:**
```bash
aws ecr get-login-password --region eu-west-3 | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com
```

### Deploy échoue: "No tasks found"

**Cause:** ECS cluster/service pas créé

**Fix:** Créer ECS cluster et service manuellement ou via Terraform/CloudFormation (voir `docs/DEPLOYMENT.md`)

### Health checks fail

**Cause:** Environment variables manquantes

**Fix:**
```bash
# Vérifier les secrets
aws secretsmanager get-secret-value --secret-id rag-system/prod/config

# Vérifier les ECS task environment
aws ecs describe-task-definition --task-definition streamlit-ui-prod
```

---

## Monitoring

### Logs en temps réel

```bash
# Via CloudWatch
aws logs tail /ecs/rag-pipeline --follow

# Via GitHub Actions
gh run watch
```

### Métriques

Créer dashboard CloudWatch avec:
- Deployment frequency
- Success rate
- Rollback count
- Deploy duration

---

## Coûts AWS Estimés

| Service | Usage | Coût mensuel |
|---------|-------|--------------|
| ECR | 2 repos, 10 images each | ~$2 |
| ECS Fargate | 2 tasks, t3.small | ~$30 |
| EC2 (Ollama) | 1 t3.medium | ~$30 |
| ALB | 1 load balancer | ~$20 |
| CloudWatch | Logs + metrics | ~$10 |
| **Total** | | **~$92/mois** |

---

## Next Steps

- [ ] Setup monitoring dashboard
- [ ] Configure alarms CloudWatch
- [ ] Setup Slack notifications
- [ ] Document runbooks
- [ ] Schedule nightly tests
- [ ] Setup cost alerts

---

## Support

**Docs:**
- [Deployment Guide](./DEPLOYMENT.md)
- [Troubleshooting](./DEPLOYMENT.md#troubleshooting)

**Contacts:**
- DevOps team: #rag-devops
- Alerts: #rag-system-alerts
