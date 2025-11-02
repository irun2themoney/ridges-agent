#!/usr/bin/env python3
"""
Upload Ridges Agent to SN62

This script uploads the bounty-optimized agent to the Ridges subnet (SN62) on Bittensor.

Prerequisites:
- btcli installed and configured
- Wallet with registered hotkey on SN62
- Sufficient balance for registration/upload

Usage:
    python3 upload_agent.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Configuration
AGENT_PATH = Path(__file__).parent / "agents/top_agent"
AGENT_FILE = AGENT_PATH / "agent.py"
NETUID = 62
NETWORK = "ridges"
WALLET_NAME = "default"
HOTKEY_NAME = "default"

def check_prerequisites():
    """Verify all prerequisites are met."""
    print("=" * 80)
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 80)
    
    # Check btcli
    try:
        result = subprocess.run(["btcli", "--version"], capture_output=True, text=True)
        print(f"✅ btcli found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ btcli not found. Please install: pip install bittensor")
        return False
    
    # Check agent file
    if not AGENT_FILE.exists():
        print(f"❌ Agent file not found: {AGENT_FILE}")
        return False
    print(f"✅ Agent file exists: {AGENT_FILE}")
    
    # Check file size
    size_kb = AGENT_FILE.stat().st_size / 1024
    print(f"✅ Agent size: {size_kb:.1f} KB")
    
    # Check wallet
    wallet_path = Path.home() / ".bittensor" / "wallets" / WALLET_NAME
    if not wallet_path.exists():
        print(f"⚠️  Wallet not found: {wallet_path}")
        print(f"   You may need to create it: btcli wallet create --wallet.name {WALLET_NAME}")
        return False
    print(f"✅ Wallet exists: {WALLET_NAME}")
    
    return True

def show_agent_info():
    """Display agent information."""
    print("\n" + "=" * 80)
    print("📊 AGENT INFORMATION")
    print("=" * 80)
    
    # Count lines
    with open(AGENT_FILE, 'r') as f:
        lines = len(f.readlines())
    
    print(f"File: {AGENT_FILE}")
    print(f"Lines: {lines}")
    print(f"Status: {'✅ PASS' if lines < 2000 else '❌ FAIL'} (under 2,000 lines requirement)")
    
    # Show first few lines
    print("\nFirst 10 lines of agent.py:")
    with open(AGENT_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            if i <= 10:
                print(f"  {i:3d}: {line.rstrip()}")
            else:
                break

def confirm_upload():
    """Get user confirmation before uploading."""
    print("\n" + "=" * 80)
    print("⚠️  UPLOAD CONFIRMATION")
    print("=" * 80)
    print(f"""
This will upload your agent to Ridges subnet (SN62) with:

  Wallet:   {WALLET_NAME}
  Hotkey:   {HOTKEY_NAME}
  Network:  {NETWORK}
  NetUID:   {NETUID}
  File:     {AGENT_FILE}

This action will:
  1. Register your hotkey on SN62 (if not already registered)
  2. Upload your agent code
  3. Make it live for evaluation

Continue? (yes/no): """.strip())
    
    response = input().strip().lower()
    return response in ['yes', 'y']

def upload_agent():
    """Upload agent to Ridges."""
    print("\n" + "=" * 80)
    print("📤 UPLOADING AGENT")
    print("=" * 80)
    
    try:
        # Step 1: Check hotkey registration
        print("\n1️⃣  Checking hotkey registration...")
        check_cmd = [
            "btcli", "subnet", "metagraph",
            "--netuid", str(NETUID),
            "--network", NETWORK
        ]
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
        print(f"   Metagraph check: Done")
        
        # Step 2: Register hotkey if needed
        print("\n2️⃣  Ensuring hotkey is registered...")
        register_cmd = [
            "btcli", "subnet", "register",
            "--netuid", str(NETUID),
            "--wallet.name", WALLET_NAME,
            "--wallet.hotkey", HOTKEY_NAME,
            "--network", NETWORK,
            "--yes"
        ]
        print(f"   Command: {' '.join(register_cmd)}")
        result = subprocess.run(register_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   ✅ Hotkey registered (or already registered)")
        else:
            print(f"   ⚠️  Registration status: {result.stdout}")
        
        # Step 3: Upload agent
        print("\n3️⃣  Uploading agent to Ridges...")
        upload_cmd = [
            "btcli", "subnet", "upload",
            "--netuid", str(NETUID),
            "--wallet.name", WALLET_NAME,
            "--wallet.hotkey", HOTKEY_NAME,
            "--network", NETWORK,
            str(AGENT_FILE),
            "--yes"
        ]
        print(f"   Command: {' '.join(upload_cmd[:5])} ...")
        
        result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("   ✅ Agent uploaded successfully!")
            print(f"\n{result.stdout}")
            return True
        else:
            print(f"   ❌ Upload failed")
            print(f"   Error: {result.stderr}")
            print(f"   Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return False

def show_success():
    """Show success message."""
    print("\n" + "=" * 80)
    print("🎉 UPLOAD COMPLETE")
    print("=" * 80)
    print(f"""
Your agent has been uploaded to Ridges SN62!

Next steps:
  1. Monitor the leaderboard at https://www.ridges.ai
  2. Watch for your agent's pass rate
  3. Check evaluation results in your account

Wallet:  {WALLET_NAME}
Hotkey:  {HOTKEY_NAME}
Agent:   {AGENT_FILE}

Your agent is now competing! 🚀

Target: >55% pass rate for bounty qualification
Size:   {AGENT_FILE.stat().st_size / 1024:.1f} KB (under 2,000 lines limit ✅)
""".strip())

def show_error():
    """Show error message."""
    print("\n" + "=" * 80)
    print("❌ UPLOAD FAILED")
    print("=" * 80)
    print("""
The upload failed. Possible reasons:
  1. btcli not installed or not in PATH
  2. Wallet not properly configured
  3. Network connection issues
  4. Invalid agent file

Troubleshooting:
  - Run: btcli wallet list
  - Run: btcli subnet metagraph --netuid 62 --network ridges
  - Check internet connection
  - Verify agent file: python3 -c "from agents.top_agent.agent import agent_main; print('✅ Agent imports OK')"
""".strip())

def main():
    """Main upload flow."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  🚀 RIDGES AGENT UPLOAD SCRIPT".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Cannot proceed with upload.")
        sys.exit(1)
    
    # Step 2: Show agent info
    show_agent_info()
    
    # Step 3: Get confirmation
    if not confirm_upload():
        print("\n❌ Upload cancelled by user.")
        sys.exit(0)
    
    # Step 4: Upload
    success = upload_agent()
    
    # Step 5: Show result
    if success:
        show_success()
        sys.exit(0)
    else:
        show_error()
        sys.exit(1)

if __name__ == "__main__":
    main()
