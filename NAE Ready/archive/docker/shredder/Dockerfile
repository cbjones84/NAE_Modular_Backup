# ================================
# Dockerfile for Shredder v4 Agent
# ================================
FROM python:3.10-slim

WORKDIR /app
COPY ../../agents/shredder.py /app/shredder.py

RUN pip install requests
RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1
CMD ["python3", "/app/shredder.py"]
NAE_AGENT_NAME=Shredder
NAE_ROLE=WealthAllocator

NAE_BITCOIN_WALLET="btc_wallet_address_here"
NAE_TAX_ACCOUNT="tax_account_identifier_here"
NAE_WEALTH_ACCOUNT="wealth_account_identifier_here"

NAE_LOG_PATH="/app/logs/shredder.log"
NAE_ENV=production

