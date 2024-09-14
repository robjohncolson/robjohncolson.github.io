const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const port = 3000;
const filePath = './passRequests.json';  // Path to the JSON file for persistence

app.use(express.json());
app.use(express.static('public'));  // Serve static files from the 'public' folder

let passRequests = [];

// Load pass requests from JSON file when the server starts
const loadPassRequests = () => {
    if (fs.existsSync(filePath)) {
        const data = fs.readFileSync(filePath);
        passRequests = JSON.parse(data);
    } else {
        passRequests = [];
    }
};

// Save pass requests to the JSON file
const savePassRequests = () => {
    fs.writeFileSync(filePath, JSON.stringify(passRequests, null, 2));
};

// Call the function to load requests when the server starts
loadPassRequests();

// Endpoint to submit a new pass request (POST)
app.post('/pass-request', (req, res) => {
    const { studentName, passType } = req.body;
    const newPassRequest = {
        id: passRequests.length + 1,  // Unique ID
        studentName: studentName,
        passType: passType,
        status: 'Pending',  // All new requests start as 'Pending'
        timestamp: new Date().toISOString(),
    };

    passRequests.push(newPassRequest);
    savePassRequests();  // Save the updated pass requests to the file
    res.status(201).json({ message: 'Pass request submitted successfully', request: newPassRequest });
});

// Endpoint to get all pass requests (GET)
app.get('/pass-requests', (req, res) => {
    res.json(passRequests);
});

// Endpoint to approve or deny a pass request (PATCH)
app.patch('/pass-request/:id', (req, res) => {
    const requestId = parseInt(req.params.id);
    const { status } = req.body;  // 'Approved' or 'Denied'

    const requestIndex = passRequests.findIndex(r => r.id === requestId);
    if (requestIndex === -1) {
        return res.status(404).json({ message: 'Pass request not found' });
    }

    // Update the status of the request
    passRequests[requestIndex].status = status;
    savePassRequests();  // Save the updated pass requests to the file
    res.json({ message: `Pass request ${status.toLowerCase()}`, request: passRequests[requestIndex] });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
