# IMPORTANT: Displays pipeline version on Bedrock UI
version = "1.0"

# "export PATH=~/miniconda/bin:$PATH"
train {
    step train {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh",
            "bash ~/miniconda.sh -b -p ~/miniconda",
            "rm ~/miniconda.sh",
            "/root/miniconda/bin/conda env update -f environment.yaml",
            "/root/miniconda/bin/conda activate veritas"
        ]
        script = [{sh = ["python train.py"]}]
        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }
}

