"""Constants for account management."""

# Tag on each member account mapping it to an environment ID
ENVIRONMENT_ID_TAG_KEY = "EnvironmentId"

# Tag on member accounts provisioned for a containerized scenario.
# Value format: "<scenario-name>/<account-tag>" — case-sensitive, never normalized.
SCENARIO_ACCOUNT_TAG_KEY = "aws-bench:scenario"

# Default role created by Organizations in every child account
ORG_ACCESS_ROLE = "OrganizationAccountAccessRole"

# IAM role CloudFormation assumes for stack deletions. Created at provisioning,
# SCP-protected. Bland name to avoid revealing the account's purpose to agents.
CFN_OPS_ROLE_NAME = "cfn-service-execution"

# Error codes a freshly-vended member account returns until its STS/IAM converges, none
# retried by botocore. Lives in this dep-free leaf so both the fast-scan engine (shipped to
# the Lambda closure) and the host retry classifier in utils/retry can import it.
FRESH_ACCOUNT_TRANSIENT_CODES = frozenset(
    {"OptInRequired", "SubscriptionRequiredException", "InvalidClientTokenId"}
)

# Tag on member accounts storing a hash of the scenario config at deploy time.
# Presence means cleanup has NOT been run yet (removed after successful cleanup).
SCENARIO_SHA_TAG_KEY = "aws-bench:scenario-sha"

# Tag on member accounts whose post-run reset failed to restore baseline. Presence
# blocks env setup and the benchmark run on that account until a successful reset or
# a clean cleanup removes it. Value is always "true"; gates check presence only.
CONTAMINATED_TAG_KEY = "aws-bench:contaminated"

# Retries for the contamination tag set/clear org calls. Organizations tagging is
# throttled and eventually-consistent, so a transient failure is retried with
# jittered exponential backoff before the caller decides the outcome.
CONTAMINATION_TAG_MAX_ATTEMPTS = 3

# Account creation polling
POLL_TIMEOUT_SEC = 300

# Region opt-in wait during env init. Enablement can outlast it; rerunning env init resumes.
REGION_OPT_IN_TIMEOUT_SEC = 1800
REGION_OPT_IN_POLL_INTERVAL_SEC = 10

# Regions AWS enables in every account. Every later region is opt-in and starts DISABLED.
DEFAULT_ENABLED_REGIONS = frozenset(
    {
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-south-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "ca-central-1",
        "eu-central-1",
        "eu-north-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    }
)

# Retries for account creation on EMAIL_ALREADY_EXISTS. Each attempt regenerates
# the email (only per-second timestamp entropy), so waits must exceed a second.
EMAIL_COLLISION_MAX_ATTEMPTS = 5
