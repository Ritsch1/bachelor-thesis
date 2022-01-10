import subprocess

subprocess.run(f"conda env create -n bachelor_arbeit -f ba_env.yml", shell=True)
print("----------------------------------------------------------------------")
print("You can now run the command 'conda activate bachelor_arbeit' and use this conda environment to run the main - notebook.")