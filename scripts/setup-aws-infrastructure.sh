#!/bin/bash

set -e

AWS_REGION=${AWS_REGION:-eu-west-3}
PROJECT_NAME="rag-system"
ENVIRONMENT=${ENVIRONMENT:-dev}

echo "========================================"
echo "AWS Infrastructure Setup"
echo "========================================"
echo "Project: $PROJECT_NAME"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo ""

create_s3_buckets() {
    echo "Creating S3 buckets..."

    BUCKET_NAME="$PROJECT_NAME-deployment-logs-$ENVIRONMENT"

    aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || echo "Bucket already exists"

    aws s3api put-bucket-encryption \
        --bucket $BUCKET_NAME \
        --server-side-encryption-configuration '{
          "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
              "SSEAlgorithm": "AES256"
            }
          }]
        }'

    aws s3api put-bucket-versioning \
        --bucket $BUCKET_NAME \
        --versioning-configuration Status=Enabled

    echo "✓ S3 bucket created: $BUCKET_NAME"
}

create_ecr_repositories() {
    echo ""
    echo "Creating ECR repositories..."

    for repo in rag-pipeline streamlit-ui; do
        aws ecr describe-repositories \
            --repository-names $repo \
            --region $AWS_REGION &> /dev/null || \
        aws ecr create-repository \
            --repository-name $repo \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256

        aws ecr put-lifecycle-policy \
            --repository-name $repo \
            --region $AWS_REGION \
            --lifecycle-policy-text '{
              "rules": [{
                "rulePriority": 1,
                "description": "Keep last 10 images",
                "selection": {
                  "tagStatus": "any",
                  "countType": "imageCountMoreThan",
                  "countNumber": 10
                },
                "action": {
                  "type": "expire"
                }
              }]
            }'

        echo "✓ Repository created: $repo"
    done
}

create_cloudwatch_log_groups() {
    echo ""
    echo "Creating CloudWatch Log Groups..."

    for log_group in \
        "/ecs/$PROJECT_NAME-rag-pipeline" \
        "/ecs/$PROJECT_NAME-streamlit-ui" \
        "/ec2/$PROJECT_NAME-ollama"; do

        aws logs create-log-group \
            --log-group-name $log_group \
            --region $AWS_REGION || echo "Log group already exists"

        aws logs put-retention-policy \
            --log-group-name $log_group \
            --retention-in-days 30 \
            --region $AWS_REGION

        echo "✓ Log group created: $log_group"
    done
}

create_iam_github_actions_role() {
    echo ""
    echo "Creating IAM role for GitHub Actions..."

    ROLE_NAME="GitHubActionsRole"

    TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_ORG/YOUR_REPO:*"
        }
      }
    }
  ]
}
EOF
)

    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "Role for GitHub Actions CI/CD" || echo "Role already exists"

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMFullAccess

    echo "✓ IAM role created: $ROLE_NAME"
    echo "  NOTE: Update trust policy with your GitHub org/repo!"
}

create_secrets_manager_secrets() {
    echo ""
    echo "Creating Secrets Manager secrets..."

    SECRET_NAME="$PROJECT_NAME/$ENVIRONMENT/config"

    aws secretsmanager create-secret \
        --name $SECRET_NAME \
        --description "RAG System configuration for $ENVIRONMENT" \
        --secret-string '{
          "AWS_BEARER_TOKEN_BEDROCK": "YOUR_TOKEN_HERE",
          "ANTHROPIC_API_KEY": "YOUR_KEY_HERE"
        }' \
        --region $AWS_REGION || echo "Secret already exists"

    echo "✓ Secret created: $SECRET_NAME"
    echo "  NOTE: Update secret values manually!"
}

setup_github_oidc_provider() {
    echo ""
    echo "Setting up GitHub OIDC provider..."

    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

    aws iam create-open-id-connect-provider \
        --url https://token.actions.githubusercontent.com \
        --client-id-list sts.amazonaws.com \
        --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 || echo "OIDC provider already exists"

    echo "✓ GitHub OIDC provider configured"
}

show_summary() {
    echo ""
    echo "========================================"
    echo "Infrastructure Setup Summary"
    echo "========================================"
    echo ""
    echo "✓ S3 buckets created"
    echo "✓ ECR repositories created"
    echo "✓ CloudWatch log groups created"
    echo "✓ IAM roles created"
    echo "✓ Secrets Manager configured"
    echo "✓ GitHub OIDC provider setup"
    echo ""
    echo "Next Steps:"
    echo "1. Update GitHub OIDC trust policy with your org/repo"
    echo "2. Update Secrets Manager with real credentials"
    echo "3. Configure GitHub Secrets:"
    echo "   - AWS_ACCOUNT_ID"
    echo "   - DOMAIN_NAME"
    echo "   - DEV_EC2_INSTANCE_ID"
    echo "   - STAGING_EC2_INSTANCE_ID"
    echo "   - PROD_EC2_INSTANCE_ID"
    echo "   - BEDROCK_TOKEN"
    echo "   - SLACK_WEBHOOK_URL"
    echo "4. Deploy ECS clusters and services"
    echo "5. Setup ALB and Route53"
}

main() {
    setup_github_oidc_provider
    create_s3_buckets
    create_ecr_repositories
    create_cloudwatch_log_groups
    create_iam_github_actions_role
    create_secrets_manager_secrets
    show_summary

    echo ""
    echo "✓ Infrastructure setup complete!"
}

main
