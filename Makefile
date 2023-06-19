.PHONY: run clean

VENV = venv
PYTHON = $(VENV)/bin/python3
STREAMLIT = $(VENV)/bin/streamlit
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	$(STREAMLIT) run app.py

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
