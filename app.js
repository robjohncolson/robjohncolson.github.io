const socket = new WebSocket('https://c881-73-16-30-214.ngrok-free.app');  // Update to use Ngrok URL

let blockchain = [];
let currentUser = '';
let viewedBlocks = new Set();  // To track which blocks have been viewed by the teacher

// Log WebSocket connection status
socket.onopen = function() {
    console.log('WebSocket connection established');
    alert('Connected to the server. Blockchain sync will start soon.');
};

// Handle incoming messages from the WebSocket server
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);

    console.log('Received message:', data);  // Debug log

    if (data.type === 'sync') {
        // Synchronize the blockchain
        blockchain = data.blockchain;
        alert('Blockchain synced successfully');
        displayPassRequests();  // Call the function to display requests after sync
    } else if (data.type === 'new_block') {
        // Add the new block to the local blockchain
        blockchain.push(data.block);
        displayPassRequests();  // Call the function to update the displayed requests
    } else if (data.type === 'block_viewed') {
        // Mark the block as viewed by the teacher
        viewedBlocks.add(data.blockIndex);
        displayPassRequests();  // Update the view for all clients
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
            displayPassRequests();  // Call the function to display requests when the teacher logs in
        } else {
            document.getElementById('student-view').style.display = 'block';  // Show student view
        }
    } else {
        alert('Please enter your name to login');
    }
});

// Function to submit a pass request (Student)
document.getElementById('submit-pass').addEventListener('click', () => {
    const passType = document.getElementById('pass-type').value;

    if (passType) {
        // Create a new block with the proper data structure
        const newBlock = {
            index: blockchain.length,
            timestamp: new Date().toISOString(),
            data: {
                studentName: currentUser,   // Ensure the student's name is correctly populated
                passType: passType,         // Ensure the pass type is properly selected
                status: 'Pending',          // Set status as Pending initially
            },
            previousHash: blockchain.length ? blockchain[blockchain.length - 1].hash : '0',
            hash: (blockchain.length + new Date().toISOString() + passType).toString()
        };

        blockchain.push(newBlock);  // Add the new block to the local blockchain

        // Send the new block to the WebSocket server
        socket.send(JSON.stringify({ type: 'new_block', block: newBlock }));
        console.log('Pass request submitted:', newBlock);  // Debug log

        displayPassRequests();  // Call the function to update the displayed requests
    } else {
        alert('Please select a pass type');
    }
});

// Function to display pass requests
function displayPassRequests() {
    const passList = document.getElementById(currentUser.toLowerCase() === 'teacher' ? 'pass-list' : 'student-pass-list');
    passList.innerHTML = '';  // Clear the list

    blockchain.forEach((block, index) => {
        const listItem = document.createElement('li');
        
        // Ensure we correctly access block.data and its properties
        const studentName = block.data.studentName || 'Unknown Student';
        const passType = block.data.passType || 'Unknown Type';
        const status = block.data.status || 'Unknown Status';
        let viewedStatus = viewedBlocks.has(index) ? 'Viewed by teacher' : 'Not viewed';

        listItem.textContent = `${studentName} requested ${passType} at ${new Date(block.timestamp).toLocaleTimeString()} - Status: ${status} - ${viewedStatus}`;

        // Teacher can mark the pass request as viewed
        if (currentUser.toLowerCase() === 'teacher' && status === 'Pending') {
            const viewButton = document.createElement('button');
            viewButton.textContent = 'Mark as Viewed';
            viewButton.addEventListener('click', () => viewPassRequest(index));
            listItem.appendChild(viewButton);
        }

        passList.appendChild(listItem);
    });
}

// Function for the teacher to view pass requests
function viewPassRequest(blockIndex) {
    // Mark this block as viewed
    socket.send(JSON.stringify({ type: 'view_block', blockIndex: blockIndex }));
    viewedBlocks.add(blockIndex);
    console.log(`Block ${blockIndex} marked as viewed by teacher`);
    displayPassRequests();  // Update the view for the teacher and students
}
