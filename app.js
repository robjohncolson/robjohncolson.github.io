// Create a WebSocket connection
const socket = new WebSocket('ws://localhost:8080');  // Ensure this URL matches your WebSocket server

let blockchain = [];
let currentUser = '';

// Log WebSocket connection status
socket.onopen = function() {
    console.log('WebSocket connection established');
};

// Handle errors with the WebSocket connection
socket.onerror = function(error) {
    console.error('WebSocket Error:', error);
};

// Handle incoming messages from the WebSocket server
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);

    console.log('Received message:', data);  // Debug log

    if (data.type === 'sync') {
        // Synchronize the blockchain
        blockchain = data.blockchain;
        displayPassRequests();
    } else if (data.type === 'new_block') {
        // Add the new block to the local blockchain
        blockchain.push(data.block);
        displayPassRequests();
    }
};

// Handle login
document.getElementById('login-button').addEventListener('click', () => {
    currentUser = document.getElementById('name').value.trim();

    if (currentUser) {
        document.getElementById('login-section').style.display = 'none';  // Hide login section
        console.log('User logged in as:', currentUser);  // Debug log

        if (currentUser.toLowerCase() === 'teacher') {
            document.getElementById('teacher-view').style.display = 'block';  // Show teacher view
            displayPassRequests();
        } else {
            document.getElementById('student-view').style.display = 'block';  // Show student view
        }
    } else {
        alert('Please enter your name to login');
        console.log('No name entered');
    }
});

// Function to submit a pass request (Student)
document.getElementById('submit-pass').addEventListener('click', () => {
    const passType = document.getElementById('pass-type').value;

    if (passType) {
        const newBlock = {
            index: blockchain.length,
            timestamp: new Date().toISOString(),
            data: {
                studentName: currentUser,
                passType: passType,
                status: 'Pending',
            },
            previousHash: blockchain.length ? blockchain[blockchain.length - 1].hash : '0',
            hash: (blockchain.length + new Date().toISOString() + passType).toString()
        };

        blockchain.push(newBlock);

        // Send the new block to the WebSocket server
        socket.send(JSON.stringify({ type: 'new_block', block: newBlock }));
        console.log('Pass request submitted:', newBlock);  // Debug log

        displayPassRequests();
    } else {
        alert('Please select a pass type');
    }
});

// Display pass requests (Teacher and Student views)
function displayPassRequests() {
    const passList = document.getElementById(currentUser.toLowerCase() === 'teacher' ? 'pass-list' : 'student-pass-list');
    passList.innerHTML = '';  // Clear the list

    blockchain.forEach(block => {
        const listItem = document.createElement('li');
        listItem.textContent = `${block.data.studentName} requested ${block.data.passType} at ${new Date(block.timestamp).toLocaleTimeString()} - Status: ${block.data.status}`;
        passList.appendChild(listItem);
    });
}
