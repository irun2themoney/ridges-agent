#!/usr/bin/env python3
"""
Direct agent upload script - bypasses CLI dependencies
"""
import hashlib
import httpx
from bittensor import Keypair
from pathlib import Path

# Configuration
API_URL = "https://platform-v2.ridges.ai"
AGENT_FILE = Path("agents/top_agent/agent.py")
WALLET_NAME = "default"
HOTKEY_NAME = "default"

def load_keypair():
    """Load keypair from Bittensor wallet"""
    try:
        from bittensor import wallet as bt_wallet
        wallet = bt_wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
        keypair = wallet.hotkey
        return keypair
    except Exception as e:
        print(f"Error loading keypair: {e}")
        print(f"Make sure wallet '{WALLET_NAME}' with hotkey '{HOTKEY_NAME}' exists")
        raise

def upload_agent():
    """Upload agent to Ridges platform"""
    print("=" * 70)
    print("RIDGES AGENT UPLOAD")
    print("=" * 70)
    
    # Load keypair
    print(f"\nLoading wallet: {WALLET_NAME}/{HOTKEY_NAME}")
    keypair = load_keypair()
    print(f"✓ Hotkey loaded: {keypair.ss58_address}")
    
    # Verify agent file
    if not AGENT_FILE.exists():
        print(f"\n✗ Agent file not found: {AGENT_FILE}")
        return False
    
    print(f"\n✓ Agent file found: {AGENT_FILE}")
    file_size = AGENT_FILE.stat().st_size
    print(f"  Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Read agent file
    print("\nReading agent file...")
    with open(AGENT_FILE, 'rb') as f:
        file_content = f.read()
    
    # Compute hash
    content_hash = hashlib.sha256(file_content).hexdigest()
    public_key = keypair.public_key.hex()
    
    # Check for existing agent
    print(f"\nChecking for existing agent...")
    print(f"API URL: {API_URL}")
    
    with httpx.Client(timeout=30.0) as client:
        # Get existing agent (if any)
        try:
            response = client.get(
                f"{API_URL}/retrieval/agent-by-hotkey",
                params={"miner_hotkey": keypair.ss58_address},
                timeout=30.0
            )
            
            if response.status_code == 200 and response.json():
                latest_agent = response.json()
                name = latest_agent.get("name")
                version_num = latest_agent.get("version_num", -1) + 1
                print(f"  ✓ Found existing agent: {name} (v{latest_agent.get('version_num', 0)})")
                print(f"  → Uploading as version {version_num}")
            else:
                name = "top-agent"  # Default name
                version_num = 0
                print(f"  ℹ No existing agent found")
                print(f"  → Creating new agent: {name} (v{version_num})")
        except Exception as e:
            print(f"  ⚠ Could not check existing agent: {e}")
            name = "top-agent"
            version_num = 0
            print(f"  → Creating new agent: {name} (v{version_num})")
        
        # Prepare upload
        print(f"\nPreparing upload...")
        file_info = f"{keypair.ss58_address}:{content_hash}:{version_num}"
        signature = keypair.sign(file_info).hex()
        
        payload = {
            'public_key': public_key,
            'file_info': file_info,
            'signature': signature,
            'name': name
        }
        
        files = {'agent_file': ('agent.py', file_content, 'text/plain')}
        
        # Upload
        print(f"\nUploading agent...")
        print(f"  Hotkey: {keypair.ss58_address}")
        print(f"  Name: {name}")
        print(f"  Version: {version_num}")
        
        try:
            response = client.post(
                f"{API_URL}/upload/agent",
                files=files,
                data=payload,
                timeout=120.0
            )
            
            if response.status_code == 200:
                print("\n" + "=" * 70)
                print("✅ UPLOAD SUCCESSFUL")
                print("=" * 70)
                print(f"\nAgent '{name}' (v{version_num}) uploaded successfully!")
                print(f"\nYour agent is now in the screening queue.")
                print(f"Monitor progress at: https://cave.ridges.ai")
                return True
            else:
                print("\n" + "=" * 70)
                print("✗ UPLOAD FAILED")
                print("=" * 70)
                try:
                    error = response.json().get('detail', 'Unknown error')
                    print(f"\nError: {error}")
                except:
                    print(f"\nHTTP {response.status_code}: {response.text}")
                return False
                
        except httpx.TimeoutException:
            print("\n✗ Upload timeout - request took too long")
            return False
        except Exception as e:
            print(f"\n✗ Upload error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = upload_agent()
    exit(0 if success else 1)

