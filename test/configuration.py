# Environment Variable Configuration
TEST_ENV_CONFIG = {
    # ModelRunner test config
    "AWS_DEFAULT_REGION": "us-west-2",
    "WORKERS": "4",
    "WORKERS_PER_CPU": "1",
    "JOB_TABLE": "TEST-JOB-TABLE",
    "ENDPOINT_TABLE": "TEST-ENDPOINT-STATS-TABLE",
    "FEATURE_TABLE": "TEST-FEATURE-TABLE",
    "REGION_REQUEST_TABLE": "TEST-REGION-REQUEST-TABLE",
    "IMAGE_QUEUE": "TEST-IMAGE-QUEUE",
    "REGION_QUEUE": "TEST-REGION-QUEUE",
    "IMAGE_STATUS_TOPIC": "TEST-IMAGE-STATUS-TOPIC",
    # Fake cred info for MOTO
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "SM_SELF_THROTTLING": "true",
}
# Fake account ID
TEST_ACCOUNT = "123456789123"

# DDB Configurations
TEST_JOB_TABLE_KEY_SCHEMA = [{"AttributeName": "image_id", "KeyType": "HASH"}]
TEST_JOB_TABLE_ATTRIBUTE_DEFINITIONS = [{"AttributeName": "image_id", "AttributeType": "S"}]

TEST_ENDPOINT_TABLE_KEY_SCHEMA = [{"AttributeName": "endpoint", "KeyType": "HASH"}]
TEST_ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "endpoint", "AttributeType": "S"},
]

TEST_FEATURE_TABLE_KEY_SCHEMA = [
    {"AttributeName": "hash_key", "KeyType": "HASH"},
    {"AttributeName": "range_key", "KeyType": "RANGE"},
]
TEST_FEATURE_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "hash_key", "AttributeType": "S"},
    {"AttributeName": "range_key", "AttributeType": "S"},
]

TEST_REGION_REQUEST_TABLE_KEY_SCHEMA = [
    {"AttributeName": "region_id", "KeyType": "HASH"},
    {"AttributeName": "image_id", "KeyType": "RANGE"},
]
TEST_REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "region_id", "AttributeType": "S"},
    {"AttributeName": "image_id", "AttributeType": "S"},
]

# S3 Configurations
TEST_RESULTS_BUCKET = "test-results-bucket"
TEST_IMAGE_FILE = "./test/data/small.ntf"
TEST_IMAGE_BUCKET = "test-image-bucket"
TEST_IMAGE_KEY = "small.ntf"
TEST_S3_FULL_BUCKET_PATH = "s3://test-results-bucket/test/data/small.ntf"

TEST_RESULTS_STREAM = "test-results-stream"

TEST_IMAGE_ID = "test-image-id"
TEST_JOB_ID = "test-job-id"
TEST_REGION_ID = "test-region-id"

TEST_ELEVATION_DATA_LOCATION = "s3://TEST-BUCKET/ELEVATION-DATA-LOCATION"
