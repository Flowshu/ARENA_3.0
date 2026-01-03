# install uv (https://docs.astral.sh/uv/getting-started/installation/)
# curl -LsSf https://astral.sh/uv/install.sh | sh

# create environment & install dependencies
uv python install 3.11
uv init
uv venv --python 3.11
source .venv/bin/activate
uv add --requirements requirements.txt
uv sync
