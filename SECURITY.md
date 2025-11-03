# Security & Wallet Protection

## 🔐 Wallet Security

### Your Wallet Location

Your Bittensor wallet is stored **locally** on your machine at:
```
~/.bittensor/wallets/default/hotkeys/default
```

### Security Status

✅ **WALLET FILES**: NOT in repository  
✅ **SECRETS**: NOT in code or git history  
✅ **API KEYS**: Using environment variables (not hardcoded)  
✅ **GIT IGNORE**: Environment files excluded (.env, .venv)

## ✅ What's Protected

### Repository Contents

The repository contains:
- ✅ Agent code (1,958 lines)
- ✅ Test files
- ✅ Documentation
- ✅ Configuration templates

The repository does NOT contain:
- ❌ Wallet files or private keys
- ❌ Secrets or API keys
- ❌ .env files with credentials
- ❌ Any sensitive authentication data

### Environment Variables

The agent uses environment variables for configuration:
- `SANDBOX_PROXY_URL`: Set at runtime (not hardcoded)
- `AGENT_TIMEOUT`: Configuration variable
- `RUN_ID`: Runtime identifier
- `REPO_PATH`: Repository path

**No secrets are hardcoded in the agent code.**

## 🛡️ Security Best Practices

### Wallet Protection

1. **Never commit wallet files** to git
2. **Keep wallet directory permissions restricted**
3. **Use strong passphrases** for wallet encryption
4. **Don't share your hotkey private key** with anyone
5. **Backup your wallet** in a secure location

### Code Security

1. **Use environment variables** for API keys
2. **Don't hardcode secrets** in source code
3. **Review git commits** before pushing
4. **Use .gitignore** to exclude sensitive files
5. **Rotate API keys** if accidentally exposed

### Deployment Security

1. **Verify agent code** before uploading
2. **Check for accidental secrets** in code
3. **Use secure upload channels** (HTTPS)
4. **Monitor agent performance** after deployment
5. **Report security issues** immediately

## ⚠️ Security Checklist

Before committing or pushing code:

- [ ] No wallet files in repository
- [ ] No API keys hardcoded
- [ ] No secrets in source code
- [ ] Environment variables used for sensitive data
- [ ] .gitignore excludes .env and wallet files
- [ ] Code review completed

## 🔍 Verification Commands

### Check for Secrets in Repository

```bash
# Check for wallet files
git ls-files | grep -i "wallet\|key\|secret"

# Check git history
git log --all -- "*wallet*" "*key*" "*secret*"

# Check for hardcoded secrets
grep -r "SANDBOX_PROXY_URL\|CHUTES_API_KEY\|API.*KEY" --include="*.py"
```

## 📞 Security Issues

If you discover a security issue:

1. **Immediately revoke** exposed credentials
2. **Remove secrets** from git history if needed
3. **Update** environment variables
4. **Report** to Ridges support if necessary

## ✅ Current Status

**Your wallet is secure:**
- Wallet files are NOT in the repository
- No secrets are hardcoded in code
- Environment variables are used for configuration
- Agent code is safe to publish publicly

**Your agent is secure:**
- No credentials in source code
- Uses secure environment variables
- Complies with Ridges security requirements
- Safe for public repository sharing

