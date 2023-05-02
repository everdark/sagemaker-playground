FILE_PATH = .

.PHONY: lint install-dep model-package

lint:
	@black $(FILE_PATH)
	@isort --profile black $(FILE_PATH)
	@flake8 --max-line-length 88 $(FILE_PATH)
	@bandit $(FILE_PATH)

install-dep:
	@pip install -r requirements.txt

model-package:
	tar -czf model.tar.gz models mms_user_module.py
