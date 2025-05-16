
## Directory Structure

```
/
├── prostate_cancer/TCGAslides2/        # Input SVS files directory
├── script2/                           # Generated scripts directory
│   ├── logs/                         # Job log files location
│   ├── 000.sh, 001.sh, ...           # Individual job scripts
│   └── run_all.sh                    # Master job submission script
└── Tissuelab-Model-Zoo/               # StarDist code repository
    └── nuclei_segmentation/StarDist/
        └── main_mac.py               # Main script for nuclei segmentation
```

## Configuration

Edit the following settings in the job generation script:

```python
# Path to SVS files directory
folder = "prostate_cancer/TCGAslides2"  

# Folder where scripts will be saved
script_dir = "script2"  

# Number of commands per script (can be adjusted for job granularity)
chunk_size = 1  

# Memory requested for each job (in MB)
requested_memory = 16000  

# Queue for job submission
master_job_queue = "gpu"  

# Path to main processing script
file_main_path = '/project/zhihuanglab/tiancheng/tissuelab_agent_experiments/Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/main_mac.py'

# Path to Python environment
python_env_path = '~/miniconda3/envs/TISSUE/bin/python'
```

## Usage

1. Update the configuration variables in the script to match your environment
2. Run the script to generate the job scripts:

```bash
python generate_jobs.py
```

3. Submit all jobs to the cluster using the master script:

```bash
./script2/run_all.sh
```

## Output


- Input: `prostate_cancer/TCGAslides2/slide1.svs`
- Output: `prostate_cancer/TCGAslides2/slide1.svs.h5`

## Logs

All job logs will be stored in the `script2/logs/` directory with the following naming convention:
- Standard output: `preprocess_XX.out`
- Standard error: `preprocess_XX.err`
- Aggregate logs: `XXX_agg.out` and `XXX_agg.err`

## Job Control

- To check job status: `bjobs`
- To kill all submitted jobs: `bkill 0`
- To view detailed job information: `bjobs -l <job_id>`

## Troubleshooting

- Check the error logs in `script2/logs/` if jobs fail
- Ensure paths in the configuration are correct for your system
- Verify that the Python environment has all necessary dependencies
- Confirm that the SVS files are accessible and not corrupted

## License

[Specify license information here]

## Acknowledgements

This work is part of the Tissuelab-Model-Zoo project by the Zhi Huang Lab.
