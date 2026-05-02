FROM python:3.11-slim

# Node.js cho Hardhat
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && apt-get clean

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node deps
COPY package.json .
RUN npm install

# Project files
COPY . .

# Compile contracts
RUN npm run compile

EXPOSE 8545 8080

CMD ["bash", "-c", "npm run node & sleep 3 && npm run deploy && npm run fund && echo Ready"]
