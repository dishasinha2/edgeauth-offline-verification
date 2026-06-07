# AWS Deployment Guide

## Prerequisites
- AWS CLI configured: `aws configure`
- SAM CLI installed: `pip install aws-sam-cli`
- Region: ap-south-1 (Mumbai) — set in samconfig.toml

## Deploy Steps

1. Build the Lambda package:
   ```
   cd backend/aws_serverless
   sam build
   ```

2. Deploy (first time — guided):
   ```
   sam deploy --guided
   ```
   This will create:
   - WorkforceVerificationLogs (DynamoDB)
   - WorkforceOrganizations (DynamoDB)
   - WorkforceEmployees (DynamoDB)
   - wvp-employee-media (S3)
   - wvp-models (S3)
   - WvpFunction (Lambda)
   - WvpHttpApi (API Gateway)

3. After deploy, copy the API Gateway URL from outputs:
   ```
   Key: WvpApiUrl
   Value: https://XXXXXXXX.execute-api.ap-south-1.amazonaws.com/
   ```
   Set this as WVP_AWS_API_ENDPOINT in your .env

4. Get the API key:
   ```
   aws apigateway get-api-keys --include-values --region ap-south-1
   ```
   Set this as WVP_API_KEY in your .env

5. Update the sync engine on the edge device:
   ```
   export WVP_AWS_API_ENDPOINT=https://XXXXXXXX.execute-api.ap-south-1.amazonaws.com/prod/sync
   export WVP_API_KEY=your-key-here
   ```

## Verify Deployment

Test POST /organizations:
```bash
curl -X POST https://YOUR_URL/organizations \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"name": "Test Org", "region": "APAC"}'
```
Expected: HTTP 201 with organization_id

Test POST /sync (empty batch — should return 422):
```bash
curl -X POST https://YOUR_URL/sync \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"device_id": "test", "logs": []}'
```
Expected: HTTP 422 with validation error
