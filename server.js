const WebSocket = require('ws');
const Blockchain = require('./blockchain');  // Assuming blockchain logic is in a separate file

// Create a WebSocket server on port 8080
const wss = new WebSocket.Server({ port: 8080 });
let blockchain = new Blockchain();

wss.on('connection', (ws) => {
    console.log('A student or teacher connected');

    // Send the entire blockchain to the newly connected client
    ws.send(JSON.stringify({ type: 'sync', blockchain: blockchain.chain }));

    // Listen for messages from students/teacher
    ws.on('message', (message) => {
        const data = JSON.parse(message);

        if (data.type === 'new_block') {
            // Add a new block for the student's pass request
            const newBlock = data.block;
            blockchain.addBlock(newBlock);
            console.log('New block added to the blockchain:', newBlock);

            // Broadcast the new block to all other clients
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({ type: 'new_block', block: newBlock }));
                }
            });
        } else if (data.type === 'view_block') {
            // Teacher viewed the pass request
            console.log('Teacher viewed block:', data.blockIndex);

            // Broadcast the view event to all clients, especially the student
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({ type: 'block_viewed', blockIndex: data.blockIndex }));
                }
            });
        }
    });
});
