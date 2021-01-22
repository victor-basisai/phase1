# IMPORTANT: Displays pipeline version on Bedrock UI
version = "1.0"

# "export PATH=~/miniconda/bin:$PATH"
train {
    step train {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "echo hello world",
            "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh",
            "echo downloaded miniconda",
            "bash ~/miniconda.sh -b -p ~/miniconda",
            "echo installed miniconda",
            "rm ~/miniconda.sh",
            "echo attempting conda env",
            "/root/miniconda/bin/conda env update -f environment.yaml",
            "echo attempting conda activate",
            "/root/miniconda/bin/conda activate veritas",
            "echo installation completed"
        ]
        script = [{sh = ["python train.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }
}

