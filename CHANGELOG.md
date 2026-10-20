# Changelog

## [1.0.0](https://github.com/aws-bench/aws-bench/compare/v0.7.0...v1.0.0) (2026-10-20)


### Features

* **agents:** Bedrock bearer-token support for opencode ([#55](https://github.com/aws-bench/aws-bench/issues/55)) ([bc346f7](https://github.com/aws-bench/aws-bench/commit/bc346f76a1ee076db142a7ac2d5f0f5c02edd851))
* **aws:** support pre-existing accounts ([#26](https://github.com/aws-bench/aws-bench/issues/26)) ([e008295](https://github.com/aws-bench/aws-bench/commit/e008295e7b961cf3105e38f09829d4001712379a))
* **codex:** add marketplace and plugin support ([#81](https://github.com/aws-bench/aws-bench/issues/81)) ([8919f9c](https://github.com/aws-bench/aws-bench/commit/8919f9ccfccd11f3c3c9851244525c16c222137d))
* **dataset:** list local tasks grouped by scenario ([#102](https://github.com/aws-bench/aws-bench/issues/102)) ([13014a3](https://github.com/aws-bench/aws-bench/commit/13014a3a22c4985a7845292357d13c089b5024d5))
* **general:** Merging integration-testing branch into main ([#60](https://github.com/aws-bench/aws-bench/issues/60)) ([51e467e](https://github.com/aws-bench/aws-bench/commit/51e467e2f609715b18c89ca11cb70c3d8a96c236))
* **preflight:** verify Docker Compose v2 and buildx plugins ([#59](https://github.com/aws-bench/aws-bench/issues/59)) ([ee3ccaf](https://github.com/aws-bench/aws-bench/commit/ee3ccaf6e20306e6bd1c7519e9fa7657cbe5e685))
* **task:** refresh credentials during task phases ([#78](https://github.com/aws-bench/aws-bench/issues/78)) ([b571f1c](https://github.com/aws-bench/aws-bench/commit/b571f1cc86cfb52633ca28b5015862aae8e61da3))


### Bug Fixes

* **bedrock:** Remove duplicate bearer-token script ([#39](https://github.com/aws-bench/aws-bench/issues/39)) ([594bbeb](https://github.com/aws-bench/aws-bench/commit/594bbeb2fc293e964721a6d31deae5688835d684))
* **cleanup:** add AWS::IoT::Policy custom delete handler ([#63](https://github.com/aws-bench/aws-bench/issues/63)) ([a0bc4bf](https://github.com/aws-bench/aws-bench/commit/a0bc4bfcf07f63c0143713d63322732303b1fc9d))
* **cleanup:** add SageMaker custom handler for resources not supported by CCAPI ([#46](https://github.com/aws-bench/aws-bench/issues/46)) ([9737152](https://github.com/aws-bench/aws-bench/commit/9737152ca853d14f802bfd6d96246a6b25b2f5fc))
* **cleanup:** add service-API delete handlers for reset orphans ([#108](https://github.com/aws-bench/aws-bench/issues/108)) ([208e73d](https://github.com/aws-bench/aws-bench/commit/208e73dda320fdfa3682e7670c8412bb362e9a60))
* **cleanup:** clean up install-efs-csi-fips reset residuals ([#67](https://github.com/aws-bench/aws-bench/issues/67)) ([156a5ad](https://github.com/aws-bench/aws-bench/commit/156a5add50ba02e8eadce42edee64eb20244550b))
* **cleanup:** clean up managed-ad-with-ldaps reset residuals ([#72](https://github.com/aws-bench/aws-bench/issues/72)) ([f7c0ab8](https://github.com/aws-bench/aws-bench/commit/f7c0ab8e186426de8a2e3b91913027d197ceebe2))
* **cleanup:** CloudFormation DeleteStack fallback + Kinesis Data Analytics handler ([#71](https://github.com/aws-bench/aws-bench/issues/71)) ([6c2d389](https://github.com/aws-bench/aws-bench/commit/6c2d38963b6d67bd58c9831a998da9af18639462))
* **cleanup:** delete Bedrock KBs whose execution role isn't AWS-named ([#42](https://github.com/aws-bench/aws-bench/issues/42)) ([f23b54c](https://github.com/aws-bench/aws-bench/commit/f23b54c795dbbb17fea045373ee817f98d71b0a0))
* **cleanup:** delete EKS nodegroups before ASG preparation ([#68](https://github.com/aws-bench/aws-bench/issues/68)) ([12ec5cd](https://github.com/aws-bench/aws-bench/commit/12ec5cdff182702f8aabd13795070e462b1a5cf5))
* **cleanup:** disable termination protection in DeleteStack fallback ([#92](https://github.com/aws-bench/aws-bench/issues/92)) ([4755b10](https://github.com/aws-bench/aws-bench/commit/4755b104d51b916fbf3133e09b882394acb7d065))
* **cleanup:** handle v2-managed KDA app duality and Studio stack teardown ([#73](https://github.com/aws-bench/aws-bench/issues/73)) ([9c957f0](https://github.com/aws-bench/aws-bench/commit/9c957f060e2eec9f46a3d29dc5c643e86827c2dc))
* **cleanup:** make MediaLive channels, inputs and security groups deletable ([#22](https://github.com/aws-bench/aws-bench/issues/22)) ([b84a1ec](https://github.com/aws-bench/aws-bench/commit/b84a1ec32561589797aac67e953eee570db3fa5f))
* **cleanup:** Reclaim Lake Formation governed Glue databases ([#47](https://github.com/aws-bench/aws-bench/issues/47)) ([3b4383e](https://github.com/aws-bench/aws-bench/commit/3b4383e3b988dccea9d894ce45b21cd1933b36b1))
* **cleanup:** resolve compute-and-data resource leak failures ([#29](https://github.com/aws-bench/aws-bench/issues/29)) ([ccee19b](https://github.com/aws-bench/aws-bench/commit/ccee19bdee6420a0c36aacf77207154fc5f4053b))
* **cleanup:** stop redundant CCAPI verification in orphan sweeps ([#56](https://github.com/aws-bench/aws-bench/issues/56)) ([ea4b3bb](https://github.com/aws-bench/aws-bench/commit/ea4b3bb58ea45d8ecd9c9be7634d5135fa66bff2))
* **cleanup:** wait for EKS nodegroup deletion ([#66](https://github.com/aws-bench/aws-bench/issues/66)) ([9228a03](https://github.com/aws-bench/aws-bench/commit/9228a036dd46fd28212296897e31ba9906c6b9de))
* **codex:** Emit stdio MCP command and args as separate TOML keys ([#61](https://github.com/aws-bench/aws-bench/issues/61)) ([8748fd2](https://github.com/aws-bench/aws-bench/commit/8748fd26ef42909ec28db75da9e9f9403e99167a))
* **credentials:** reuse cached Bedrock key on transient verify failures ([#70](https://github.com/aws-bench/aws-bench/issues/70)) ([c2e8c1b](https://github.com/aws-bench/aws-bench/commit/c2e8c1b323d3a5b2bd49ca6aa55986b1b45a04ac))
* **fastscan:** list only custom GameLift locations; filter AWS ones as managed ([#84](https://github.com/aws-bench/aws-bench/issues/84)) ([74e2b38](https://github.com/aws-bench/aws-bench/commit/74e2b385b6d871fdfcbf7661852d79ff61e8478c))
* **fastscan:** mark Translate ParallelData unavailable where not offered ([#101](https://github.com/aws-bench/aws-bench/issues/101)) ([818dd20](https://github.com/aws-bench/aws-bench/commit/818dd20151a90e5e1ebee82253789a813dfd6def))
* Fix initialization in disabled regions and during SCP propagation ([#82](https://github.com/aws-bench/aws-bench/issues/82)) ([1dece1c](https://github.com/aws-bench/aws-bench/commit/1dece1c1eea62f87133761979ef3475f61ee9bbd))
* **init:** enable opt-in regions from the management account ([#88](https://github.com/aws-bench/aws-bench/issues/88)) ([34e7324](https://github.com/aws-bench/aws-bench/commit/34e7324d79b1973c59bea0749d07e97b959dd4f8))
* **init:** re-land management-account region opt-in with lazy Account Management calls ([#91](https://github.com/aws-bench/aws-bench/issues/91)) ([542037a](https://github.com/aws-bench/aws-bench/commit/542037acd5a63287cd0ba8182d864952cc0a1841))
* **init:** reconcile region SCPs before snapshots ([#75](https://github.com/aws-bench/aws-bench/issues/75)) ([1e52a44](https://github.com/aws-bench/aws-bench/commit/1e52a440e36abe4d45ba76d0f6e1e1ce8a3b7532))
* **init:** revert enable opt-in regions from the management account ([#88](https://github.com/aws-bench/aws-bench/issues/88)) ([#90](https://github.com/aws-bench/aws-bench/issues/90)) ([952f6c9](https://github.com/aws-bench/aws-bench/commit/952f6c93d307ed0060e57f0b9e9a3d9fe8715628))
* **kiro:** add kiro skills to their correct path so the headless default agent loads them ([#52](https://github.com/aws-bench/aws-bench/issues/52)) ([602e0b6](https://github.com/aws-bench/aws-bench/commit/602e0b6c4915ae689ba199658d44373b96f67e15))
* **kiro:** add support v2/v3 engine trajectories + pin --agent-engine v3 ([#79](https://github.com/aws-bench/aws-bench/issues/79)) ([ea65432](https://github.com/aws-bench/aws-bench/commit/ea65432b5ce1d838b932728fcec4e02f493761a5))
* **kiro:** make MCP servers usable on the v3 engine ([#83](https://github.com/aws-bench/aws-bench/issues/83)) ([b2a85b7](https://github.com/aws-bench/aws-bench/commit/b2a85b7a20596cd62bcbbf983db3467de394e33e))
* **kiro:** resolve MCP tool names in v3 (KAS) trajectories ([#87](https://github.com/aws-bench/aws-bench/issues/87)) ([014a604](https://github.com/aws-bench/aws-bench/commit/014a60432fd31bc4276743e659aad493adcfe589))
* **listers:** disable tnb:* listers due to deprecation ([#86](https://github.com/aws-bench/aws-bench/issues/86)) ([c8c202c](https://github.com/aws-bench/aws-bench/commit/c8c202cc2e8bd59d145cfc62e45506de32d4fd17))
* **registry:** add missing extra_instruction_paths for 'all' dataset ([#65](https://github.com/aws-bench/aws-bench/issues/65)) ([59d7f7c](https://github.com/aws-bench/aws-bench/commit/59d7f7ccbb96cab6a357bf682521292c5edcb728))
* **reset:** delete new CloudFormation stacks whole, before the individual sweep ([#89](https://github.com/aws-bench/aws-bench/issues/89)) ([cfa4788](https://github.com/aws-bench/aws-bench/commit/cfa4788e27f933f613c4dff0c8ded8c9f0ec4de9))
* **reset:** uniquify per-trial reset container name ([#54](https://github.com/aws-bench/aws-bench/issues/54)) ([7e4079b](https://github.com/aws-bench/aws-bench/commit/7e4079b8fb437fe73169cbcf117bae2f966a7280))
* **verify:** exclude AWS-reserved ElastiCache default subnet group ([#64](https://github.com/aws-bench/aws-bench/issues/64)) ([7da4521](https://github.com/aws-bench/aws-bench/commit/7da4521d6293f049ba8f6b38e81e2aac68e29666))
* **verify:** exclude DynamoDB import-from-S3 job records ([#69](https://github.com/aws-bench/aws-bench/issues/69)) ([20450d6](https://github.com/aws-bench/aws-bench/commit/20450d6fd0b8ecd2b95ccbbc4fb9c2009bb60006))
* **verify:** surface per-lister error when a baseline type can't be enumerated ([#100](https://github.com/aws-bench/aws-bench/issues/100)) ([d990db5](https://github.com/aws-bench/aws-bench/commit/d990db5feb31b1e97c079bc96fb2d1cc662be6d3))
* **verify:** tolerate baseline Lambda auto-created log groups ([#103](https://github.com/aws-bench/aws-bench/issues/103)) ([7f5a331](https://github.com/aws-bench/aws-bench/commit/7f5a33138a9f2ff21c14545d03fd13b208d39eee))


### Refactoring

* **credentials:** neutralize STS session names end-to-end ([#57](https://github.com/aws-bench/aws-bench/issues/57)) ([d50476e](https://github.com/aws-bench/aws-bench/commit/d50476e0c6e10303fa473d9df5fb8a8cc7db6d88))
* **credentials:** neutralize STS session-name prefix ([#51](https://github.com/aws-bench/aws-bench/issues/51)) ([5fb2bcb](https://github.com/aws-bench/aws-bench/commit/5fb2bcb94e736b75edbbff92d2f4fe0b21643d08))

## v0.7.0 (2026-07-24)

### Features

- **ci**: add GitHub Actions workflows, dependabot, and docs config (#4)
- **ci**: run make check as a Github Action (#3)
- **github workflow**: add Dependabot workflow

### Bug Fixes

- **ci**: skip PR Slack notification for Dependabot (#11)
- **ci**: set least-privilege permissions in notification workflows (#7)
- **ci**: prevent code injection in pr-notification workflow (#6)
- **ci**: prevent code injection in issue-notification workflow (#5)
- **deps**: apply transitive dependency version pinning
