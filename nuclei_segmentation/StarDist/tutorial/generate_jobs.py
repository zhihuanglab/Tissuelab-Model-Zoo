import os
import math

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
folder = "prostate_cancer/TCGAslides2"  # Path to SVS files
script_dir = "script2"  # Folder where scripts will be saved
chunk_size = 1          # Number of commands per script
requested_memory = 16000  # Memory per job (in MB)
master_job_queue = "gpu"  # Queue used for submitting the master script
file_main_path = '/project/zhihuanglab/tiancheng/tissuelab_agent_experiments/Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/main_mac.py'
python_env_path = '~/miniconda3/envs/TISSUE/bin/python'

# Ensure the script directory exists
os.makedirs(script_dir, exist_ok=True)
os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)

# Collect all commands
commands = []

# ------------------------------------------------------------------------------
# 1) GATHER COMMANDS
# ------------------------------------------------------------------------------
for filename in os.listdir(folder):
    if filename.lower().endswith(".svs"):
        fullpath = os.path.join(folder, filename)
        
        # Remove the leading "/project/zhihuanglab/tiancheng/" part
        #############CHANGE IF IT IS NEEDED################################
        slidepath = fullpath.replace("/project/zhihuanglab/tiancheng/", "")

        # Convert "TCGAslides" -> "TCGA_result" and remove ".svs"
        stardist_dir = slidepath.replace("TCGAslides2", "TCGA_result").replace(".svs", "")
        #update your python env and file path
        cmd = (
            #f'~/miniconda3/envs/TISSUE/bin/python /project/zhihuanglab/tiancheng/tissuelab_agent_experiments/Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/main_mac.py '
            #f'--slidepath "{slidepath}" '
            f'"{python_env_path}" "{file_main_path}" '
            f'--slidepath "{slidepath}" '
            
        )
        commands.append(cmd)

# ------------------------------------------------------------------------------
# 2) SPLIT COMMANDS INTO MULTIPLE SCRIPTS
# ------------------------------------------------------------------------------
num_scripts = math.ceil(len(commands) / chunk_size)
script_paths = []

for i in range(num_scripts):
    script_name = os.path.join(script_dir, f"{i:03d}.sh")  # "script1/00.sh", "script1/01.sh"
    with open(script_name, "w") as script_file:
        # Write the shebang
        script_file.write("#!/bin/bash\n\n")

        # Write LSF/BSUB directives
        script_file.write(f"#BSUB -J preprocess_{i:02d}\n")
        script_file.write(f"#BSUB -o {script_dir}/logs/preprocess_{i:02d}.out\n")
        script_file.write(f"#BSUB -e {script_dir}/logs/preprocess_{i:02d}.err\n")
        #script_file.write(f"#BSUB -q {queue_name}\n")
        #script_file.write(f'#BSUB -R "rusage[mem={requested_memory}]" \n')
        #script_file.write(f'#BSUB -M {requested_memory} \n')
        script_file.write("#load_openslide\n")
        script_file.write("export PYTHONUNBUFFERED=1\n\n")

        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(commands))
        chunk = commands[start_idx:end_idx]

        for cmd in chunk:
            script_file.write(cmd + "\n")  # Removed 'bsub' as requested

    # Make script executable
    os.chmod(script_name, 0o755)
    script_paths.append(script_name)

# ------------------------------------------------------------------------------
# 3) CREATE A MASTER SCRIPT THAT SUBMITS ALL SCRIPTS
# ------------------------------------------------------------------------------
master_script = os.path.join(script_dir, "run_all.sh")
with open(master_script, "w") as master_file:
    master_file.write("#!/bin/bash\n\n")
    for sp in script_paths:
        log_out = os.path.join(script_dir, "logs", os.path.basename(sp).replace(".sh", "_agg.out"))
        log_err = os.path.join(script_dir, "logs", os.path.basename(sp).replace(".sh", "_agg.err"))
        master_file.write(
            f"bsub "
            f"-q {master_job_queue} "
            f"-o {log_out} -e {log_err} "
            
            f"sh {sp}\n"
        )

# Make the master script executable
os.chmod(master_script, 0o755)

# Final output
print(f"Generated {num_scripts} chunk script(s) in '{script_dir}'")
print(f"Master script: {master_script}")
print("To submit all jobs, run:")
print(f"  ./{master_script}")