# variables
PYTHON=python3
PIP=pip

# default target
help:
	@echo "Available commands:"
	@echo "make train"
	@echo "make run"
	@echo "make install"
	@echo "make clean"

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) main.py --mode train

run:
	$(PYTHON) main.py

commit:
	make clean
	git add .
	git commit 
	git push origin 

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
