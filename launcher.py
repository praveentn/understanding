# launcher.py - Simple launcher for Windows users
import os
import sys
import subprocess

def clear_screen():
    """Clear screen (Windows compatible)"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print welcome banner"""
    print("🧠" + "=" * 58 + "🧠")
    print("   ENHANCED UNDERSTANDING FRAMEWORK")
    print("   Human-Like Learning with Forgetting")
    print("🧠" + "=" * 58 + "🧠")
    print()

def print_menu():
    """Print main menu"""
    print("What would you like to do?")
    print()
    print("1. 🚀 Quick Demo (5 minutes)")
    print("   Test the basic system with forgetting")
    print()
    print("2. 🧠 Full Training (30-60 minutes)")  
    print("   Complete human-like learning evaluation")
    print()
    print("3. ⚙️  Custom Training")
    print("   Choose your own settings")
    print()
    print("4. 📊 Evaluate Existing Model")
    print("   Test a previously trained model")
    print()
    print("5. 🔧 Setup/Install Packages")
    print("   Install required packages")
    print()
    print("6. 📁 View Results")
    print("   Open results folder")
    print()
    print("7. ❌ Exit")
    print()

def run_python_script(script_args):
    """Run Python script with arguments"""
    try:
        cmd = [sys.executable] + script_args
        print(f"Running: {' '.join(cmd)}")
        print("-" * 50)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        print("-" * 50)
        if process.returncode == 0:
            print("✅ Completed successfully!")
        else:
            print(f"❌ Failed with return code {process.returncode}")
            
        return process.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def quick_demo():
    """Run quick demo"""
    print("🚀 Starting Quick Demo...")
    print("This will take about 5 minutes and test basic forgetting mechanisms.")
    print()
    
    input("Press Enter to start, or Ctrl+C to cancel...")
    return run_python_script(["main.py", "--mode", "quick_demo"])

def full_training():
    """Run full training"""
    print("🧠 Starting Full Training...")
    print("This will take 30-60 minutes and run comprehensive evaluation.")
    print("The system will:")
    print("  - Train model with human-like forgetting")
    print("  - Test forgetting curves (Ebbinghaus)")  
    print("  - Test interference effects")
    print("  - Test consolidation benefits")
    print("  - Generate detailed reports")
    print()
    
    response = input("Continue? (y/n): ").lower().strip()
    if response == 'y':
        return run_python_script(["main.py", "--mode", "full_training"])
    else:
        print("Cancelled.")
        return False

def custom_training():
    """Run custom training with user settings"""
    print("⚙️  Custom Training Setup")
    print()
    
    try:
        print("Enter your settings (press Enter for defaults):")
        
        iterations = input("Number of iterations (default 500): ").strip()
        if not iterations:
            iterations = "500"
        
        d_model = input("Model dimension (default 256): ").strip()
        if not d_model:
            d_model = "256"
            
        num_classes = input("Number of classes (default 5): ").strip()
        if not num_classes:
            num_classes = "5"
            
        inner_lr = input("Inner learning rate (default 0.01): ").strip()
        if not inner_lr:
            inner_lr = "0.01"
            
        outer_lr = input("Outer learning rate (default 0.001): ").strip() 
        if not outer_lr:
            outer_lr = "0.001"
        
        print()
        print(f"Settings: {iterations} iters, d_model={d_model}, classes={num_classes}")
        print(f"         inner_lr={inner_lr}, outer_lr={outer_lr}")
        print()
        
        response = input("Start training? (y/n): ").lower().strip()
        if response == 'y':
            args = [
                "main.py", "--mode", "custom", 
                "--iters", iterations,
                "--d_model", d_model,
                "--num_classes", num_classes, 
                "--inner_lr", inner_lr,
                "--outer_lr", outer_lr
            ]
            return run_python_script(args)
        else:
            print("Cancelled.")
            return False
            
    except KeyboardInterrupt:
        print("\nCancelled.")
        return False

def evaluate_model():
    """Evaluate existing model"""
    print("📊 Evaluate Existing Model")
    print()
    
    # Look for existing models
    model_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pth") and "model" in file.lower():
                model_files.append(os.path.join(root, file))
    
    if model_files:
        print("Found these model files:")
        for i, model_file in enumerate(model_files):
            print(f"  {i+1}. {model_file}")
        print()
        
        try:
            choice = input("Enter number to select, or path to model file: ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    model_path = model_files[idx]
                else:
                    print("Invalid selection.")
                    return False
            else:
                model_path = choice
                
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
                
            return run_python_script(["main.py", "--mode", "evaluation_only", "--model_path", model_path])
            
        except ValueError:
            print("Invalid input.")
            return False
    else:
        print("No model files found.")
        model_path = input("Enter path to model file: ").strip()
        if model_path and os.path.exists(model_path):
            return run_python_script(["main.py", "--mode", "evaluation_only", "--model_path", model_path])
        else:
            print("Model file not found.")
            return False

def setup_packages():
    """Setup and install packages"""
    print("🔧 Package Setup")
    print()
    
    return run_python_script(["setup.py"])

def view_results():
    """Open results folder"""
    print("📁 Results Folders:")
    print()
    
    # Find results directories
    results_dirs = []
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("results_"):
            results_dirs.append(item)
    
    if results_dirs:
        results_dirs.sort(reverse=True)  # Most recent first
        print("Found these results:")
        for i, results_dir in enumerate(results_dirs):
            print(f"  {i+1}. {results_dir}")
        print()
        
        try:
            choice = input("Enter number to open folder (or Enter for most recent): ").strip()
            
            if not choice:
                folder = results_dirs[0]
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(results_dirs):
                    folder = results_dirs[idx]
                else:
                    print("Invalid selection.")
                    return False
            else:
                print("Invalid input.")
                return False
                
            # Open folder in file explorer (Windows)
            os.startfile(folder)
            print(f"Opened {folder} in file explorer.")
            return True
            
        except Exception as e:
            print(f"Error opening folder: {e}")
            return False
    else:
        print("No results folders found. Run training first.")
        return False

def main():
    """Main launcher loop"""
    while True:
        try:
            clear_screen()
            print_banner()
            print_menu()
            
            choice = input("Enter choice (1-7): ").strip()
            
            clear_screen()
            print_banner()
            
            if choice == "1":
                quick_demo()
            elif choice == "2":
                full_training()
            elif choice == "3":
                custom_training()
            elif choice == "4":
                evaluate_model()
            elif choice == "5":
                setup_packages()
            elif choice == "6":
                view_results()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
            
            print()
            input("Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()

