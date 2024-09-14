// Function to submit a pass request (Student)
function submitPassRequest(studentName, passType) {
    const timestamp = new Date().toISOString();
    const newPassRequest = {
        studentName: studentName,
        passType: passType,
        status: 'Pending',
        timestamp: timestamp
    };

    const newBlock = new Block(passChain.chain.length, timestamp, newPassRequest);
    passChain.addBlock(newBlock);
    console.log('New pass request submitted:', newBlock);
}

// Function for the teacher to approve or deny requests
function updatePassRequestStatus(blockIndex, status) {
    let block = passChain.chain[blockIndex];
    if (block && block.data.status === 'Pending') {
        block.data.status = status;
        block.hash = block.calculateHash();
        console.log(`Pass request at index ${blockIndex} updated to: ${status}`);
    } else {
        console.log('Invalid or already processed request.');
    }
}

// Display the current state of the blockchain
function displayPassRequests() {
    passChain.chain.forEach(block => {
        console.log(block);
    });
}
