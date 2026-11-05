# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately**. Do not open public GitHub issues, discussions, or pull requests for them.

Use GitHub's private vulnerability reporting:

**[Report a vulnerability](https://github.com/arpieb/ragas-ng/security/advisories/new)** (repository → *Security* tab → *Report a vulnerability*)

This opens a private advisory visible only to you and the maintainers, where we can discuss, fix, and coordinate disclosure.

Not every detail is required, but these help us triage quickly:

- **Type of issue** (e.g. SSRF, path traversal, code execution, credential leak)
- **Affected version / commit**
- **Affected files and location in code**
- **Configuration** needed to reproduce
- **Reproduction steps** and, if possible, a proof of concept
- **Impact**: what an attacker could achieve
- **Suggested fix**, if you have one

## Supported Versions

ragas-ng has not cut a release yet. Security fixes land on the `main` branch.

| Version | Supported |
| --- | --- |
| `main` | :white_check_mark: |
| upstream `ragas` releases (vibrantlabsai/ragas) | :x: — report those upstream |

## What to Expect

This is a community-maintained fork, so timelines are best effort:

1. We aim to acknowledge reports within a few days.
2. We investigate and confirm the issue.
3. We develop a fix, ideally validated with the reporter.
4. We publish the fix and a GitHub security advisory crediting the reporter (unless you prefer otherwise).

We follow [Coordinated Vulnerability Disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).

## References

- [GitHub: Privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
- [OpenSSF Best Practices](https://www.bestpractices.dev/)
