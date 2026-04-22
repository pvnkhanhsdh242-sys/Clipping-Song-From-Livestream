# Security Policy

## Supported Versions
This MVP is under active development. Security fixes are applied to `main`.

## Reporting a Vulnerability
Please open a private security report through GitHub Security Advisories if available.
If unavailable, open an issue with minimal exploit details and mark it as security-sensitive.

## Secrets and Credentials
- Do not commit API keys.
- Keep `ACOUSTID_API_KEY` in environment variables or GitHub secrets.
- `.env` is local-only and should stay untracked.

## External Services
- Paid music-ID APIs are intentionally excluded from core behavior.
- Optional AcoustID lookup should remain isolated and disabled by default.
