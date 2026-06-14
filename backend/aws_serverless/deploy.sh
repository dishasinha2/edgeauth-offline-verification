#!/bin/bash
# =============================================================================
# EdgeAuth AWS SAM Deployment Script
# =============================================================================
# Run from the backend/aws_serverless/ directory:
#   chmod +x deploy.sh && ./deploy.sh
#
# Prerequisites:
#   - AWS CLI configured: aws configure
#   - SAM CLI installed:  pip install aws-sam-cli
#   - AWS region: ap-south-1 (Mumbai)
# =============================================================================

set -e  # Exit immediately on any error

STACK_NAME="edgeauth-workforce-verification"
REGION="ap-south-1"
# Unique S3 bucket for SAM deployment artifacts (created fresh each deploy)
S3_DEPLOY_BUCKET="edgeauth-sam-deployments-$(date +%s)"

echo "============================================="
echo "  EdgeAuth AWS SAM Deployment"
echo "  Stack:  $STACK_NAME"
echo "  Region: $REGION"
echo "============================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create a temporary S3 bucket for SAM artifacts
# ---------------------------------------------------------------------------
echo "[1/4] Creating deployment artifact bucket: $S3_DEPLOY_BUCKET"
aws s3 mb "s3://$S3_DEPLOY_BUCKET" --region "$REGION"

# ---------------------------------------------------------------------------
# Step 2: SAM build
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Building Lambda package..."
sam build

# ---------------------------------------------------------------------------
# Step 3: SAM deploy
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Deploying stack to AWS..."
sam deploy \
  --stack-name "$STACK_NAME" \
  --s3-bucket "$S3_DEPLOY_BUCKET" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    DynamoDBTableName=WorkforceVerificationLogs \
    OrganizationsTableName=WorkforceOrganizations \
    EmployeesTableName=WorkforceEmployees \
    S3MediaBucketName=wvp-employee-media \
    S3ModelsBucketName=wvp-models \
  --no-confirm-changeset

# ---------------------------------------------------------------------------
# Step 4: Print post-deploy instructions
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Fetching stack outputs..."
API_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='WvpApiUrl'].OutputValue" \
  --output text 2>/dev/null || echo "UNAVAILABLE")

echo ""
echo "============================================="
echo "  Deployment Complete"
echo "============================================="
echo ""
echo "  API Gateway URL: $API_URL"
echo ""
echo "  Next steps:"
echo "  1. Copy the URL above and set in your .env:"
echo "     WVP_AWS_API_ENDPOINT=${API_URL}prod/sync"
echo ""
echo "  2. Get your API key:"
echo "     aws apigateway get-api-keys --include-values --region $REGION"
echo "     Then set: WVP_API_KEY=<the key value>"
echo ""
echo "  3. Restart the edge server:"
echo "     python ai_engine/server.py"
echo "============================================="
