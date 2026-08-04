# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainer via the repository's contact form or issue tracker
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue
5. If possible, suggest a fix

We aim to respond within 48 hours and will work with you to:

- Confirm the vulnerability
- Determine the severity
- Develop and release a fix
- Credit reporters (unless opted out)

## Security Principles

Langbot is an AI agent with **elevated system access**. Users assume responsibility for its operation.

### What the Agent Can Do

- Execute shell commands on the host system
- Read, write, and modify files
- Make outbound network requests
- Access secrets stored in the vault

### Built-in Protections

| Protection | Component | Description |
|------------|-----------|-------------|
| Catastrophic command blocklist | `components/safety.py` | Blocks filesystem wipes, fork bombs, raw disk destruction |
| SSRF protection | `components/web_tools.py` | Prevents DNS rebinding and internal network access |
| Encrypted vault | `components/vault.py` | Stores credentials with AES-256 encryption |
| Blast radius warnings | `components/routing.py` | Warns before destructive operations |
| Memory isolation | `MEMORY_POLICY.md` | All state stored in `./memory/` directory |

### Important Warnings

> **⚠️ This software grants an AI agent unrestricted shell, file, and network access on the host that runs it. You assume all risk arising from running it.**

1. **Run in isolated environments** — Use containers, VMs, or sandboxed environments
2. **Limit permissions** — Run with minimal required privileges
3. **Network isolation** — Restrict outbound connections as needed
4. **Monitor activity** — Log and review agent actions
5. **Secure the vault** — Use strong master passwords; never commit vault files

## Credential Management

### Vault

The vault (`./memory/vault/`) stores encrypted credentials:

- Master key protected with 0600 permissions
- Credentials encrypted at rest
- Never commit vault files to version control

### Best Practices

- Use unique credentials per service
- Rotate secrets regularly
- Never log or display secret values
- Use environment variables for runtime overrides

## Dependency Security

- Dependencies are pinned in `requirements.txt`
- Run `pip audit` regularly to check for vulnerabilities
- Review third-party code before adding dependencies
- Keep dependencies updated

## Configuration Security

### Required Settings

```json
{
  "paths": {
    "vault_dir": "./memory/vault/"
  },
  "safety": {
    "enable_catastrophic_blocklist": true
  },
  "http_request": {
    "allowed_networks": ["10.0.0.0/8", "172.16.0.0/12"],
    "block_internal": true
  }
}
```

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `LANGBOT_VAULT_PASSWORD` | Vault master key | Yes (if using vault) |
| `LANGBOT_CONFIG` | Config file path | No |
| `AGENT_SCRATCH_DIR` | Scratch directory | No |
| `AGENT_CHROMA_DIR` | Memory directory | No |

**Never commit `.env` files or config files containing secrets.**

## Security Headers

When deploying langbot behind a web server:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'
```

## Security Updates

Security patches are released as minor version updates. Subscribe to:

- GitHub Releases notifications
- Repository watch (releases only)

## Scope

This security policy covers:

- ✅ Core langbot functionality
- ✅ Built-in components (`components/`)
- ✅ Vault encryption
- ✅ SSRF protection
- ✅ Catastrophic command blocking

This policy does **NOT** cover:

- ❌ Third-party dependencies
- ❌ User-provided configurations
- ❌ Environment-specific deployments
- ❌ Model providers (OpenAI, Anthropic, etc.)

## License

This software is licensed for **personal use only**. See [LICENSE](../../LICENSE) for terms.
