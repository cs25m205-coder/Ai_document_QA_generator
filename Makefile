install:
	pip install -r requirements.txt

api:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

ui:
	streamlit run streamlit_app.py

test:
	pytest -q
