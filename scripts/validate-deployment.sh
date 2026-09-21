#!/bin/bash

set -e

ENVIRONMENT=${1:-dev}

echo "========================================"
echo "Pre-Deployment Validation"
echo "Environment: $ENVIRONMENT"
echo "========================================"
echo ""

ERRORS=0

check_aws_credentials() {
    echo "✓ Checking AWS credentials..."
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "  ✗ ERROR: AWS credentials not configured"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
    echo "  ✓ AWS credentials OK"
}

check_docker() {
    echo "✓ Checking Docker..."
    if ! docker info &> /dev/null; then
        echo "  ✗ ERROR: Docker not running"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
    echo "  ✓ Docker OK"
}

check_images_in_ecr() {
    echo "✓ Checking images in ECR..."

    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=${AWS_REGION:-eu-west-3}

    for repo in rag-pipeline streamlit-ui; do
        if ! aws ecr describe-images \
            --repository-name $repo \
            --region $AWS_REGION \
            --image-ids imageTag=latest &> /dev/null; then
            echo "  ✗ ERROR: Image $repo:latest not found in ECR"
            ERRORS=$((ERRORS + 1))
        else
            echo "  ✓ Image $repo:latest found in ECR"
        fi
    done
}

check_secrets() {
    echo "✓ Checking Secrets Manager..."

    SECRET_NAME="rag-system/$ENVIRONMENT/config"

    if ! aws secretsmanager describe-secret \
        --secret-id $SECRET_NAME &> /dev/null; then
        echo "  ✗ ERROR: Secret $SECRET_NAME not found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    SECRET_VALUE=$(aws secretsmanager get-secret-value \
        --secret-id $SECRET_NAME \
        --query SecretString \
        --output text)

    if ! echo "$SECRET_VALUE" | jq -e '.AWS_BEARER_TOKEN_BEDROCK' &> /dev/null; then
        echo "  ✗ ERROR: AWS_BEARER_TOKEN_BEDROCK not set in secret"
        ERRORS=$((ERRORS + 1))
    else
        echo "  ✓ Secret $SECRET_NAME OK"
    fi
}

check_ecs_cluster() {
    echo "✓ Checking ECS cluster..."

    CLUSTER_NAME="rag-${ENVIRONMENT}-cluster"

    if ! aws ecs describe-clusters \
        --clusters $CLUSTER_NAME \
        --query 'clusters[0].status' \
        --output text | grep -q "ACTIVE"; then
        echo "  ✗ ERROR: ECS cluster $CLUSTER_NAME not active"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    echo "  ✓ ECS cluster $CLUSTER_NAME active"
}

check_service_health() {
    echo "✓ Checking service health..."

    SERVICE_NAME="streamlit-ui-service-${ENVIRONMENT}"
    CLUSTER_NAME="rag-${ENVIRONMENT}-cluster"

    RUNNING_COUNT=$(aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --query 'services[0].runningCount' \
        --output text)

    DESIRED_COUNT=$(aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --query 'services[0].desiredCount' \
        --output text)

    if [ "$RUNNING_COUNT" != "$DESIRED_COUNT" ]; then
        echo "  ⚠ WARNING: Service $SERVICE_NAME - Running: $RUNNING_COUNT, Desired: $DESIRED_COUNT"
    else
        echo "  ✓ Service $SERVICE_NAME healthy ($RUNNING_COUNT/$DESIRED_COUNT tasks)"
    fi
}

check_github_secrets() {
    echo "✓ Checking GitHub Secrets..."

    REQUIRED_SECRETS=(
        "AWS_ACCOUNT_ID"
        "DOMAIN_NAME"
        "BEDROCK_TOKEN"
        "SLACK_WEBHOOK_URL"
    )

    echo "  NOTE: Cannot verify GitHub secrets from CLI"
    echo "  Please manually verify these secrets exist in GitHub:"
    for secret in "${REQUIRED_SECRETS[@]}"; do
        echo "    - $secret"
    done
}

run_tests() {
    echo "✓ Running tests..."

    if ! uv run pytest tests/ -q; then
        echo "  ✗ ERROR: Tests failed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    echo "  ✓ Tests passed"
}

check_lint() {
    echo "✓ Running lint checks..."

    if ! uv run ruff check . --quiet; then
        echo "  ✗ ERROR: Lint checks failed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    echo "  ✓ Lint checks passed"
}

show_summary() {
    echo ""
    echo "========================================"
    echo "Validation Summary"
    echo "========================================"

    if [ $ERRORS -eq 0 ]; then
        echo "✅ All checks passed!"
        echo ""
        echo "Ready to deploy to $ENVIRONMENT"
        return 0
    else
        echo "❌ $ERRORS check(s) failed"
        echo ""
        echo "Fix errors before deploying"
        return 1
    fi
}

main() {
    check_aws_credentials || true
    check_docker || true
    check_images_in_ecr || true
    check_secrets || true
    check_ecs_cluster || true
    check_service_health || true
    check_github_secrets || true
    run_tests || true
    check_lint || true

    show_summary
}

main
