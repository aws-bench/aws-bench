# Changelog

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
