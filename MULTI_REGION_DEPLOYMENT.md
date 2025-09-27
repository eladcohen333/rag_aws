# הוראות פריסה רב-אזורית: תשתית בפרנקפורט עם Bedrock בארה"ב

⚠️ **חשוב מאוד**: כאשר עוברים מאזור אחד לאחר, חובה למחוק את הסטאק הישן קודם!

## סקירה מהירה בעברית

1. **הסטאק הישן**: אם יש לך סטאק קיים ב-us-east-1, הוא יימחק אוטומטית
2. **הסטאק החדש**: יפרס בפרנקפורט (eu-central-1) 
3. **Bedrock**: יישאר בארה"ב (us-east-1) כמו עכשיו
4. **GitHub Actions**: מטפל בכל התהליך אוטומטית

---

# Multi-Region Deployment Guide: Frankfurt Infrastructure with US Bedrock

This guide explains how to deploy the AWS RAG Solution with infrastructure in Frankfurt (eu-central-1) while keeping Bedrock services in the US (us-east-1).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frankfurt (eu-central-1)                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Web Interface (CloudFront + S3)                          │ │
│  │ • API Gateway + Lambda Functions                           │ │
│  │ • DynamoDB Tables                                          │ │
│  │ • Aurora PostgreSQL (RAG Engine)                          │ │
│  │ • OpenSearch (if enabled)                                 │ │
│  │ • Kendra (if enabled)                                     │ │
│  │ • SageMaker Models (if enabled)                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                            Cross-Region Access
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                     US East (us-east-1)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Amazon Bedrock Models                                    │ │
│  │ • Bedrock Knowledge Bases                                  │ │
│  │ • Bedrock Guardrails                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Changes Made

### 1. Updated Configuration (`bin/config.json`)

```json
{
  "bedrock": {
    "enabled": true,
    "region": "us-east-1",
    "guardrails": {
      "enabled": false,
      "identifier": "",
      "version": "DRAFT"
    }
  },
  "deployment": {
    "primaryRegion": "eu-central-1",
    "bedrockRegion": "us-east-1"
  }
}
```

### 2. GitHub Actions Workflow Updates

- **Primary Region**: Changed from `us-east-1` to `eu-central-1`
- **Bedrock Region**: Remains `us-east-1`
- **Bootstrap**: Both regions are bootstrapped for CDK deployment

### 3. CDK Infrastructure Changes

- **Main Stack**: Deploys to Frankfurt (`eu-central-1`)
- **Cross-Region Access**: Lambda functions configured with `BEDROCK_REGION` environment variable
- **IAM Permissions**: Bedrock permissions allow cross-region access

### 4. Runtime Configuration

- **Bedrock Client**: Automatically uses `us-east-1` region regardless of deployment region
- **Environment Variables**: `BEDROCK_REGION=us-east-1` set for all Lambda functions
- **Cross-Region Calls**: Handled transparently by the Bedrock client

## Deployment Steps

### Prerequisites

1. **AWS Account Setup**
   - Ensure you have appropriate permissions in both regions
   - Bedrock model access enabled in `us-east-1`
   - CDK bootstrap completed in both regions

2. **GitHub Actions Setup**
   - Update the IAM role ARN in the workflow if needed
   - Ensure the role has permissions in both regions

### Step 1: Update Configuration

The configuration has already been updated in this repository. Review and modify `bin/config.json` if needed:

```bash
# Review the configuration
cat bin/config.json
```

### Step 2: Deploy via GitHub Actions

⚠️ **Important**: The workflow will automatically detect and destroy any existing stack in `us-east-1` before deploying to Frankfurt.

1. **Push to Main Branch**
   ```bash
   git add .
   git commit -m "Configure multi-region deployment: Frankfurt + US Bedrock"
   git push origin main
   ```

2. **Monitor Deployment**
   - Check GitHub Actions workflow execution
   - Verify old stack destruction (if exists) in us-east-1
   - Verify both regions are bootstrapped
   - Confirm new stack deployment in Frankfurt

### Step 3: Manual Deployment (Alternative)

If deploying manually, **you must destroy the old stack first**:

```bash
# Step 1: Destroy existing stack in us-east-1 (if exists)
export CDK_DEFAULT_REGION=us-east-1
aws cloudformation describe-stacks --region us-east-1 --stack-name RAG-GenAIChatBotStack
# If stack exists, destroy it:
npx cdk destroy RAG-GenAIChatBotStack --force --region us-east-1

# Step 2: Set environment variables for Frankfurt deployment
export CDK_DEFAULT_REGION=eu-central-1
export AWS_DEFAULT_REGION=eu-central-1
export AWS_BEDROCK_REGION=us-east-1

# Step 3: Bootstrap both regions
npx cdk bootstrap aws://YOUR_ACCOUNT_ID/eu-central-1
npx cdk bootstrap aws://YOUR_ACCOUNT_ID/us-east-1

# Step 4: Deploy the new stack in Frankfurt
npx cdk deploy --all --require-approval never
```

## Verification

### 1. Check Infrastructure Deployment

```bash
# Verify stack in Frankfurt
aws cloudformation describe-stacks \
  --region eu-central-1 \
  --stack-name RAG-GenAIChatBotStack

# Verify Bedrock access from Frankfurt
aws bedrock list-foundation-models \
  --region us-east-1
```

### 2. Test Cross-Region Functionality

1. **Access the Web Interface**
   - CloudFront distribution serves from Frankfurt
   - Backend APIs run in Frankfurt

2. **Test Bedrock Models**
   - Chat functionality should work with US Bedrock models
   - Embedding models should function correctly
   - RAG queries should work with cross-region setup

### 3. Monitor Logs

```bash
# Check Lambda function logs in Frankfurt
aws logs describe-log-groups \
  --region eu-central-1 \
  --log-group-name-prefix "/aws/lambda/RAG-"

# Monitor cross-region calls
aws logs filter-log-events \
  --region eu-central-1 \
  --log-group-name "/aws/lambda/RAG-LangchainInterface-RequestHandler" \
  --filter-pattern "BEDROCK_REGION"
```

## Cost Considerations

### Data Transfer Costs

- **Cross-Region API Calls**: Bedrock API calls from Frankfurt to US East incur data transfer charges
- **Typical Costs**: $0.02 per GB for data transfer between regions
- **Optimization**: Consider caching strategies for frequently used embeddings

### Regional Pricing Differences

- **Frankfurt vs US East**: Some services may have different pricing
- **Bedrock Pricing**: Remains the same regardless of calling region
- **Lambda Costs**: May vary slightly between regions

## Troubleshooting

### Common Issues

1. **Bedrock Access Denied**
   ```
   Solution: Verify IAM permissions include bedrock:* actions
   Check: Ensure BEDROCK_REGION environment variable is set
   ```

2. **Cross-Region Latency**
   ```
   Expected: 100-150ms additional latency for Bedrock calls
   Monitor: CloudWatch metrics for Lambda duration
   ```

3. **Model Availability**
   ```
   Issue: Some Bedrock models may not be available in us-east-1
   Solution: Check model availability and update configuration
   ```

### Debug Commands

```bash
# Check environment variables in Lambda
aws lambda get-function-configuration \
  --region eu-central-1 \
  --function-name RAG-LangchainInterface-RequestHandler \
  --query 'Environment.Variables.BEDROCK_REGION'

# Test Bedrock connectivity
aws bedrock invoke-model \
  --region us-east-1 \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --body '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":100}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

## Security Considerations

### Cross-Region Access

- **IAM Policies**: Ensure least-privilege access to Bedrock resources
- **VPC Configuration**: Consider VPC endpoints for secure communication
- **Encryption**: All data transfer is encrypted in transit

### Compliance

- **Data Residency**: Main application data stays in Frankfurt
- **AI Model Processing**: Bedrock processing occurs in US East
- **Audit Logging**: CloudTrail logs cross-region API calls

## Rollback Procedure

If you need to revert to single-region deployment:

1. **Update Configuration**
   ```json
   {
     "deployment": {
       "primaryRegion": "us-east-1",
       "bedrockRegion": "us-east-1"
     }
   }
   ```

2. **Redeploy**
   ```bash
   export CDK_DEFAULT_REGION=us-east-1
   npx cdk deploy --all
   ```

3. **Clean Up Frankfurt Resources**
   ```bash
   npx cdk destroy --region eu-central-1
   ```

## Support and Monitoring

### CloudWatch Dashboards

- Monitor cross-region latency
- Track Bedrock API call success rates
- Monitor data transfer costs

### Alerts

Set up CloudWatch alarms for:
- High cross-region latency (>500ms)
- Bedrock API errors
- Unusual data transfer volumes

## Next Steps

1. **Performance Optimization**
   - Implement caching for embeddings
   - Consider regional model deployment strategies

2. **Cost Optimization**
   - Monitor and optimize cross-region data transfer
   - Evaluate regional pricing differences

3. **Scaling Considerations**
   - Plan for increased cross-region traffic
   - Consider additional regions for global deployment