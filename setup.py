# setup.py - Pure Python setup for Windows
import os
import sys
import subprocess
import platform

def run_command(command):
    """Run command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {command}")
        print(f"Exception: {e}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} detected")
        print("   Python 3.8+ required")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0"
    ]
    
    print("📦 Installing packages...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    
    # Install packages one by one
    for package in requirements:
        print(f"Installing {package}...")
        if not run_command(f"{sys.executable} -m pip install {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        import tqdm
        print("✅ tqdm")
        
        # Test device
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ Device: {device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["results", "logs", "models", "data"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}/")
        except Exception as e:
            print(f"❌ Failed to create {directory}/: {e}")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements_content = """# Core deep learning
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
"""
    
    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("✅ Created requirements.txt")
    except Exception as e:
        print(f"❌ Failed to create requirements.txt: {e}")

def main():
    """Main setup function"""
    print("🚀 Enhanced Understanding Framework Setup")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.executable}")
    print()
    
    # Check Python version
    if not check_python():
        print("\n❌ Setup failed: Python version too old")
        input("Press Enter to exit...")
        return False
    
    # Create directories
    create_directories()
    
    # Create requirements file
    create_requirements_file()
    
    # Install packages
    print("\n📦 Installing required packages...")
    print("This may take a few minutes...")
    
    if not install_requirements():
        print("\n⚠️  Some packages failed to install")
        print("You can try installing manually:")
        print("pip install torch numpy pandas matplotlib scikit-learn tqdm")
        input("Press Enter to continue...")
    
    # Test imports
    print("\n🧪 Testing installation...")
    if test_imports():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now run:")
        print("  python main.py --mode quick_demo")
        print("  python main.py --mode full_training")
        
    else:
        print("\n❌ Setup completed with errors")
        print("Some packages may not be working correctly")
    
    print("\n" + "=" * 50)
    input("Press Enter to exit...")
    return True

if __name__ == "__main__":
    main()

