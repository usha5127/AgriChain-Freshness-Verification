AgriChain: IoT-Based Freshness Detection & Blockchain Traceability
Overview

AgriChain is an integrated framework for real-time freshness monitoring, shelf-life prediction, and transparent traceability of agricultural produce.
It combines IoT sensing, machine learning, and blockchain to enable data-driven quality verification across the supply chain.

Key Features
Real-time monitoring using IoT sensors
Verified Quality Index (VQI) for freshness scoring
Shelf-life prediction and early spoilage detection
Blockchain-based traceability (Ethereum + IPFS)
Smart contract-based quality validation
QR-based product tracking
System Architecture
IoT Sensors → Cloud → ML Models → VQI → Blockchain (IPFS + Ethereum)
Components

IoT Layer

MQ-135 (gas/VOC detection)
DHT11 (temperature & humidity)

Machine Learning Layer

Data preprocessing and feature engineering
Models: SVM, Gradient Boosting, XGBoost

Blockchain Layer

Ethereum smart contracts
IPFS (Pinata) for storage
Immutable traceability records
Verified Quality Index (VQI)

A dynamic freshness score (0–100) computed using:

VOC concentration
Temperature
Humidity
Range	Status
67–100	Fresh
34–66	Moderate
0–33	Spoiled
Results
Dataset: 8000 samples
Best model: XGBoost
Accuracy: 98.8%
Spoilage detection: 12–18 hours before visible decay
Tech Stack

Hardware

ESP8266 NodeMCU
MQ-135, DHT11

Software

Python, Flask
React.js
ThingSpeak

Blockchain

Ethereum
IPFS (Pinata)
Solidity, Web3.js
Project Structure
AgriChain/
├── hardware/
├── backend/
├── frontend/
├── ml-models/
├── blockchain/
├── dataset/
├── docs/
└── README.md
Setup
Clone
git clone https://github.com/usha5127/AgriChain-Freshness-Verification.git
cd AgriChain
Backend
cd backend
pip install -r requirements.txt
python app.py
Frontend
cd frontend
npm install
npm start
Blockchain
cd blockchain
truffle migrate
Workflow
Sensors collect environmental data
Data sent to cloud
ML computes VQI
Data stored in IPFS (CID)
CID recorded on blockchain
QR used for verification
Use Cases
Farmers: fair pricing
Suppliers: quality monitoring
Retailers: informed decisions
Consumers: transparency
Authors
Sharmili Nukapeyi
Haritha Yennu
Lakshmi Durga Tirumani
Usha Rani Yanadhi
Deepthi Annie
License

For academic and research purposes.
