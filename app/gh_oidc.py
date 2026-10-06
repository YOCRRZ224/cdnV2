"""
Verifies GitHub Actions OIDC ID tokens so the backend can trust *who* is
pushing without any shared secret living in contributor repos.

Why this instead of a PAT/App token per contributor:
- The token is minted per-workflow-run by GitHub itself and is only valid
  for ~5 minutes.
- Its `repository_owner` / `repository` / `ref` claims are set by GitHub's
  own infrastructure from the run's real context - a workflow author cannot
  edit their own workflow YAML to make GitHub sign a token claiming to be a
  different repo or a different owner.
- We never see (or need) a long-lived credential from the contributor.

Docs: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
"""

import time
import httpx
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException

GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
JWKS_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks"
EXPECTED_AUDIENCE = "cdn.amit.is-a.dev"

_jwks_client = None
_jwks_client_ts = 0
_JWKS_CLIENT_TTL = 3600


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client, _jwks_client_ts
    now = time.time()
    if _jwks_client is None or now - _jwks_client_ts > _JWKS_CLIENT_TTL:
        _jwks_client = PyJWKClient(JWKS_URL)
        _jwks_client_ts = now
    return _jwks_client


def verify_actions_token(token: str) -> dict:
    """Verify a GitHub Actions OIDC token and return its trusted claims.

    Raises HTTPException(401) if the token is missing, expired, wrongly
    scoped, or fails signature verification.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing GitHub Actions OIDC token.")

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=EXPECTED_AUDIENCE,
            issuer=GITHUB_OIDC_ISSUER,
            options={"require": ["exp", "iat", "repository", "repository_owner"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid GitHub Actions token: {e}")

    repository = claims.get("repository", "")
    owner = claims.get("repository_owner", "")
    if "/" not in repository:
        raise HTTPException(status_code=401, detail="Malformed token claims.")

    repo_owner_from_full, repo_name = repository.split("/", 1)
    if repo_owner_from_full != owner or not owner or not repo_name:
        raise HTTPException(status_code=401, detail="Token owner/repository mismatch.")

    return {
        "owner": owner,
        "owner_id": str(claims.get("repository_owner_id", "")),
        "repo": repo_name,
        "actor": claims.get("actor", ""),
        "ref": claims.get("ref", ""),
        "workflow_ref": claims.get("job_workflow_ref", ""),
    }
