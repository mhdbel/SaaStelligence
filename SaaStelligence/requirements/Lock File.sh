# Install pip-tools
pip install pip-tools

# Generate locked versions
pip-compile requirements.txt -o requirements-lock.txt

# Install from lock file
pip install -r requirements-lock.txt