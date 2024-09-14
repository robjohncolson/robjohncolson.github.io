const WebSocket = require('ws');
const Blockchain = require('./blockchain');  // Assuming blockchain logic is in a separate file

const wss = new WebSocket.Server({ port: 8080 });
let blockchain = new Blockchain();

wss.on('connection', (ws) => {
    console.log('A student connected');

    // Send the entire blockchain to the newly connected client
    ws.send(JSON.stringify({ type: 'sync', blockchain: blockchain.chain }));

    // Listen for messages from students
    ws.on('message', (message) => {
        const data = JSON.parse(message);

        if (data.type === 'new_block') {
            const newBlock = data.block;
            blockchain.addBlock(newBlock);
            console.log('New block added to the blockchain:', newBlock);

            // Broadcast the new block to all other clients
            wss.clients.forEach(client => {
                if (client !== ws && client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify({ type: 'new_block', block: newBlock }));
                }
            });
        }
    });
});
