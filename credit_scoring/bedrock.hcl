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
            "export PATH=$HOME/miniconda/bin:$PATH",
            "conda env update -f environment-train.yaml",
            "eval \"$(conda shell.bash hook)\"",
            "conda activate veritas"
        ]
        script = [{sh = ["python train.py"]}]
        resources {
            cpu = "1.0"
            memory = "4G"
        }
    }
}

serve {
    image = "continuumio/miniconda"
    install = [
        "conda env update -f environment-deploy.yaml",
        "conda activate veritas"
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve_http:app"
        ]}
    ]
    
    parameters {
        WORKERS = "1"
    }
}