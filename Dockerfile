FROM python:3.12-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /home/user

COPY --chown=user:user requirements.txt .

RUN pip install \
    --user \
    --no-cache-dir \
    --break-system-packages \
    -r requirements.txt

COPY --chown=user:user submission/ ./submission/

ENTRYPOINT ["python3", "-m", "submission.main"]