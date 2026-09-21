#!/bin/bash

set -e

AWS_REGION=${AWS_REGION:-eu-west-3}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}
IMAGE_TAG=${IMAGE_TAG:-latest}

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "ERROR: AWS_ACCOUNT_ID environment variable not set"
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "========================================"
echo "Push Images to Amazon ECR"
echo "========================================"
echo "Registry: $ECR_REGISTRY"
echo "Tag: $IMAGE_TAG"
echo ""

authenticate_ecr() {
    echo "Authenticating with ECR..."
    aws ecr get-login-password --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ECR_REGISTRY
    echo "✓ Authenticated"
}

create_repositories() {
    echo ""
    echo "Creating ECR repositories if they don't exist..."

    for repo in rag-pipeline streamlit-ui; do
        aws ecr describe-repositories \
            --repository-names $repo \
            --region $AWS_REGION &> /dev/null || \
        aws ecr create-repository \
            --repository-name $repo \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        echo "✓ Repository $repo ready"
    done
}

build_images() {
    echo ""
    echo "Building images..."

    docker build -f Dockerfile.rag-pipeline -t rag-pipeline:$IMAGE_TAG .
    docker build -f Dockerfile.streamlit -t streamlit-ui:$IMAGE_TAG .

    echo "✓ Images built"
}

tag_and_push() {
    echo ""
    echo "Tagging and pushing images..."

    for service in rag-pipeline streamlit-ui; do
        echo "  - Processing $service..."

        docker tag $service:$IMAGE_TAG $ECR_REGISTRY/$service:$IMAGE_TAG
        docker push $ECR_REGISTRY/$service:$IMAGE_TAG

        if [ "$IMAGE_TAG" != "latest" ]; then
            docker tag $service:$IMAGE_TAG $ECR_REGISTRY/$service:latest
            docker push $ECR_REGISTRY/$service:latest
        fi

        echo "  ✓ $service pushed"
    done
}

show_images() {
    echo ""
    echo "========================================"
    echo "Pushed Images"
    echo "========================================"

    for repo in rag-pipeline streamlit-ui; do
        echo ""
        echo "Repository: $repo"
        aws ecr describe-images \
            --repository-name $repo \
            --region $AWS_REGION \
            --query 'imageDetails[0:5].[imageTags[0],imagePushedAt,imageSizeInBytes]' \
            --output table
    done
}

main() {
    authenticate_ecr
    create_repositories
    build_images
    tag_and_push
    show_images

    echo ""
    echo "✓ Images successfully pushed to ECR!"
    echo ""
    echo "Images:"
    echo "  - $ECR_REGISTRY/rag-pipeline:$IMAGE_TAG"
    echo "  - $ECR_REGISTRY/streamlit-ui:$IMAGE_TAG"
}

main
