import os
import sys
import shutil
import subprocess

def create_directory_structure():
    """Create the necessary project directory structure"""
    print("Creating project directory structure...")
    
    directories = [
        'data',
        'src',
        'src/data',
        'src/models',
        'src/visualization',
        'models',
        'models/preprocessing',
        'reports',
        'reports/figures',
        'data/predictions',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files in Python package directories
        if directory.startswith('src'):
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated __init__.py file\n')

def run_seed_script():
    """Run the seed.py script to generate sample data"""
    print("Generating sample data with seed.py...")
    try:
        subprocess.run([sys.executable, 'seed.py'], check=True)
        print("✅ Sample data generated successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error running seed.py script")
        return False
    return True

def train_model():
    """Train the salary prediction model"""
    print("Training salary prediction model...")
    try:
        subprocess.run([sys.executable, 'src/train_and_save_models.py'], check=True)
        print("✅ Model trained and saved successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error training model")
        return False
    return True

def setup_project():
    """Set up the entire project"""
    print("Starting project setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Generate sample data
    if not run_seed_script():
        print("Failed to generate sample data. Setup incomplete.")
        return
    
    # Move CSV to the right location
    csv_source = "turkiye_it_sektoru_calisanlari.csv"
    csv_dest = os.path.join("data", "turkiye_it_sektoru_calisanlari.csv")
    
    if os.path.exists(csv_source):
        shutil.copy(csv_source, csv_dest)
        print(f"✅ Copied {csv_source} to {csv_dest}")
    else:
        print(f"❌ {csv_source} not found. Manual data setup required.")
    
    # Train the model
    if train_model():
        print("✅ Model training completed successfully!")
    else:
        print("❌ Model training failed. Setup incomplete.")
        return
    
    print("\nProject setup completed!")
    print(f"\nTo run the application, use:\n\n    streamlit run src/app.py\n")

if __name__ == "__main__":
    setup_project()