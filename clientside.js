const socket = new WebSocket('ws://localhost:8080');  // Connect to the teacher's device

// Handle incoming messages
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'sync') {
        // Synchronize the blockchain
        blockchain.chain = data.blockchain;
        displayStudentPassRequests();
    } else if (data.type === 'new_block') {
        // Add the new block to the local blockchain
        blockchain.addBlock(data.block);
        displayStudentPassRequests();
    }
};

// Function to submit a pass request
function submitPassRequest(studentName, passType) {
    const newBlock = new Block(blockchain.chain.length, new Date().toISOString(), {
        studentName: studentName,
        passType: passType,
        status: 'Pending',
    });

    blockchain.addBlock(newBlock);

    // Send the new block to the teacher's server
    socket.send(JSON.stringify({ type: 'new_block', block: newBlock }));

    console.log('Pass request submitted:', newBlock);
}
