## Unreleased

### Feat

- **dataset**: list local tasks grouped by scenario (#102)
- **codex**: add marketplace and plugin support (#81)
- **task**: refresh credentials during task phases (#78)
- **general**: Merging integration-testing branch into main (#60)
- **preflight**: verify Docker Compose v2 and buildx plugins (#59)
- **agents**: Bedrock bearer-token support for opencode (#55)
- **aws**: support pre-existing accounts (#26)

### Fix

- **verify**: tolerate baseline Lambda auto-created log groups (#103)
- **fastscan**: mark Translate ParallelData unavailable where not offered (#101)
- **reset**: delete new CloudFormation stacks whole, before the individual sweep (#89)
- **verify**: surface per-lister error when a baseline type can't be enumerated (#100)
- **init**: re-land management-account region opt-in with lazy Account Management calls (#91)
- **kiro**: resolve MCP tool names in v3 (KAS) trajectories (#87)
- **init**: revert enable opt-in regions from the management account (#88) (#90)
- **fastscan**: list only custom GameLift locations; filter AWS ones as managed (#84)
- **kiro**: make MCP servers usable on the v3 engine (#83)
- **init**: enable opt-in regions from the management account (#88)
- **listers**: disable tnb:* listers due to deprecation (#86)
- Fix initialization in disabled regions and during SCP propagation (#82)
- **kiro**: add support v2/v3 engine trajectories + pin --agent-engine v3 (#79)
- **init**: reconcile region SCPs before snapshots (#75)
- **cleanup**: delete EKS nodegroups before ASG preparation (#68)
- **cleanup**: clean up managed-ad-with-ldaps reset residuals (#72)
- **verify**: exclude DynamoDB import-from-S3 job records (#69)
- **kiro**: add kiro skills to their correct path so the headless default agent loads them (#52)
- **credentials**: reuse cached Bedrock key on transient verify failures (#70)
- **cleanup**: CloudFormation DeleteStack fallback + Kinesis Data Analytics handler (#71)
- **cleanup**: clean up install-efs-csi-fips reset residuals (#67)
- **cleanup**: wait for EKS nodegroup deletion (#66)
- **cleanup**: add AWS::IoT::Policy custom delete handler (#63)
- **registry**: add missing extra_instruction_paths for 'all' dataset (#65)
- **verify**: exclude AWS-reserved ElastiCache default subnet group (#64)
- **codex**: Emit stdio MCP command and args as separate TOML keys (#61)
- **cleanup**: stop redundant CCAPI verification in orphan sweeps (#56)
- **reset**: uniquify per-trial reset container name (#54)
- **cleanup**: Reclaim Lake Formation governed Glue databases (#47)
- **cleanup**: add SageMaker custom handler for resources not supported by CCAPI (#46)
- **cleanup**: delete Bedrock KBs whose execution role isn't AWS-named (#42)
- **bedrock**: Remove duplicate bearer-token script (#39)
- **cleanup**: resolve compute-and-data resource leak failures (#29)
- **cleanup**: make MediaLive channels, inputs and security groups deletable (#22)

### Refactor

- **credentials**: neutralize STS session names end-to-end (#57)
- **credentials**: neutralize STS session-name prefix (#51)

## v0.7.0 (2026-07-24)

### Feat

- **ci**: add GitHub Actions workflows, dependabot, and docs config (#4)
- **ci**: run make check as a Github Action (#3)
- **github workflow**: add Dependabot workflow

### Fix

- **ci**: skip PR Slack notification for Dependabot (#11)
- **ci**: set least-privilege permissions in notification workflows (#7)
- **ci**: prevent code injection in pr-notification workflow (#6)
- **ci**: prevent code injection in issue-notification workflow (#5)
- **deps**: apply transitive dependency version pinning
