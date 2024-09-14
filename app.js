const socket = new WebSocket('ws://random-string.ngrok.io');  // Update to use Ngrok URL

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
   
