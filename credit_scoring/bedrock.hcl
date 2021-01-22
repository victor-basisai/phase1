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
            "export PATH=/root/miniconda/bin:$PATH",
            "conda env update -f environment.yaml",
            "echo trying to source conda",
            "/bin/bash -c 'source /root/miniconda/etc/profile.d/conda.sh'",
            "echo almost completed environment setup"
        ]
        script = [{sh = ["/root/miniconda/envs/veritas/bin/python train.py"]}]
        resources {
            cpu = "1.0"
            memory = "4G"
        }
    }
}

