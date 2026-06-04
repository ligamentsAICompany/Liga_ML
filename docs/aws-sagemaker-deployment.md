# AWS SageMaker Deployment

Liga ML can run the web app anywhere, but AWS SageMaker training requires an AWS runtime identity with IAM permission to use SageMaker, S3, and CloudWatch Logs. Normalized datasets are staged to S3, the generated training code is uploaded beside the input data, SageMaker writes model artifacts back to S3, and logs are available in CloudWatch Logs.

## Required AWS Services

- SageMaker for managed training jobs.
- S3 for input data, generated code, model output, and checkpoints.
- CloudWatch Logs for job logs.
- IAM for the app runtime identity and SageMaker execution role.
- ECR if you use a custom training image.
- Secrets Manager is optional for production token storage.

## Required Environment

Set these in the app runtime environment. Prefer IAM roles over static credentials.

```bash
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket
AWS_S3_PREFIX=liga-ml
AWS_SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/LigaMLSageMakerExecutionRole
AWS_DEFAULT_INSTANCE_TYPE=ml.g5.xlarge
AWS_DEFAULT_INSTANCE_COUNT=1
AWS_DEFAULT_MAX_RUN_SECONDS=3600
AWS_OUTPUT_POLICY=aws-private
AWS_SAGEMAKER_TRAINING_IMAGE_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/your-training-image:latest
```

`AWS_SAGEMAKER_TRAINING_IMAGE_URI` is required before live submission. The app will not guess a framework image.

## IAM Roles

The app runtime identity needs permission to inspect readiness locally and submit controlled jobs:

```text
sagemaker:CreateTrainingJob
sagemaker:DescribeTrainingJob
sagemaker:ListTrainingJobs
sagemaker:StopTrainingJob
iam:PassRole
s3:GetObject
s3:PutObject
s3:ListBucket
logs:DescribeLogStreams
logs:GetLogEvents
```

The SageMaker execution role named by `AWS_SAGEMAKER_ROLE_ARN` needs:

```text
s3:GetObject
s3:PutObject
s3:ListBucket
logs:CreateLogStream
logs:PutLogEvents
ecr:GetAuthorizationToken
ecr:BatchCheckLayerAvailability
ecr:BatchGetImage
ecr:GetDownloadUrlForLayer
```

If `AWS_OUTPUT_POLICY` is `hf-hub` or `cloud-and-hf-hub`, provide `HF_TOKEN` securely through the session or a secret manager. Never print or bake tokens into images.

## S3 Layout

Each job uses the configured bucket and prefix:

```text
s3://your-s3-bucket/liga-ml/jobs/<generated-job-name>/input/train.jsonl
s3://your-s3-bucket/liga-ml/jobs/<generated-job-name>/code/train.py
s3://your-s3-bucket/liga-ml/jobs/<generated-job-name>/output/
s3://your-s3-bucket/liga-ml/jobs/<generated-job-name>/checkpoints/
```

The training result file is written into the SageMaker model directory and packaged under the output artifact.

## Training Image

`AWS_SAGEMAKER_TRAINING_IMAGE_URI` must point to an image accessible to the SageMaker execution role. The image must either contain the SFT dependencies or allow the generated script to bootstrap them with pip. ECR permissions are required for private images. Liga ML does not hard-code or infer image URIs.

## Output Policies

- `aws-private`: keep artifacts in S3 only. Recommended for sensitive domains.
- `hf-hub`: push the final model to Hugging Face Hub.
- `cloud-and-hf-hub`: keep S3 artifacts and push to Hugging Face Hub.

Use `aws-private` until data classification, model-card policy, and Hub permissions are explicitly approved.

## Validation Commands

From the repository root:

```bash
uv run python scripts/check_aws_readiness.py
uv run python scripts/aws_sagemaker_dry_run.py --dataset-name example/dataset --model-name Qwen/Qwen2.5-0.5B-Instruct --output-model-id aws-smoke-model --max-run-seconds 3600 --allow-missing-aws --allow-missing-image
curl http://localhost:5173/api/health/providers
```

`check_aws_readiness.py` may exit 1 on local machines without AWS env or credentials; the JSON output should still be safe and useful.

## First Manual Smoke Test

In the app, select AWS SageMaker and run this only after maintainers approve credentials, IAM, S3, ECR image, and cost:

```text
Run a tiny AWS SageMaker SFT smoke test using template="sft". Use dataset_name="HuggingFaceH4/ultrachat_200k", dataset_split="train_sft", model_name="Qwen/Qwen2.5-0.5B-Instruct", output_model_id="your-hf-namespace/aws-smoke-model", max_train_samples=5, max_eval_samples=2, num_train_epochs=1, max_run_seconds=3600, output_policy="aws-private", and do not push to Hugging Face Hub.
```

Confirm the approval card, cost estimate, SageMaker console URL, S3 output path, CloudWatch Logs URL, and final result markers.

## Troubleshooting

- missing AWS_S3_BUCKET or AWS_SAGEMAKER_ROLE_ARN: set the required environment variables and restart the app.
- credentials not detected: attach an IAM role in production or use local AWS config for development.
- iam:PassRole denied: allow the app runtime identity to pass only the configured SageMaker execution role.
- S3 access denied: grant the runtime and execution role access to `your-s3-bucket` and the configured prefix.
- image pull failure: confirm the ECR image exists in the selected region and the execution role can read it.
- CloudWatch logs not visible: grant log stream read permissions to the app runtime identity.
- HF_TOKEN missing: required only for `hf-hub` and `cloud-and-hf-hub` output policies.
- cost unknown: use a supported instance type or require manual approval before submission.
