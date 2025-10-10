import subprocess

def run_command(cmd):
    print(f"\n Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f" Error while running: {cmd}")
        exit(1)

if __name__ == "__main__":
    run_command("python app.py")

    if "y" == input("Do you want to run the Streamlit app to ask questions? (y/n): ").strip().lower():
        run_command("streamlit run model.py")
    
