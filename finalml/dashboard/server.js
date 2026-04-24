const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files from the dashboard folder
app.use(express.static(path.join(__dirname)));

// ML Ablation Results from SVM unified_outputs (CSV passthrough)
app.get("/api/ml/ablation-svm", (req, res) => {
  try {
    const fs = require('fs');
    const csvPath = path.join(__dirname, '..', 'svm', 'unified_outputs', 'ablation_results.csv');
    if (!fs.existsSync(csvPath)) {
      return res.status(404).json({ success: false, message: 'Ablation CSV not found', path: csvPath });
    }
    const csv = fs.readFileSync(csvPath, 'utf8');
    res.json({ success: true, type: 'csv', data: csv });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to load SVM ablation CSV', message: error.message });
  }
});

// ML Ablation Results from SVM closed (CSV passthrough)
app.get("/api/ml/ablation-closed", (req, res) => {
  try {
    const fs = require('fs');
    const csvPath = path.join(__dirname, '..', 'svm', 'closed', 'ablation_results.csv');
    if (!fs.existsSync(csvPath)) {
      return res.status(404).json({ success: false, message: 'Closed ablation CSV not found', path: csvPath });
    }
    const csv = fs.readFileSync(csvPath, 'utf8');
    res.json({ success: true, type: 'csv', data: csv });
  } catch (error) {
    res.status(500).json({ success: false, error: 'Failed to load SVM closed ablation CSV', message: error.message });
  }
});

// Serve the main dashboard
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

const PORT = 3002;
app.listen(PORT, () => {
  console.log(`🚀 Ablation Dashboard running on port ${PORT}`);
  console.log(`📊 Access at: http://localhost:${PORT}`);
});
