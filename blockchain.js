class Block {
    constructor(index, previousHash, timestamp, data, hash) {
        this.index = index;
        this.previousHash = previousHash;
        this.timestamp = timestamp;
        this.data = data;
        this.hash = hash;
    }
}

class Blockchain {
    constructor() {
        this.chain = [this.createGenesisBlock()];
    }

    createGenesisBlock() {
        return new Block(0, "0", new Date().toISOString(), "Genesis Block", this.calculateHash(0, "0", new Date().toISOString(), "Genesis Block"));
    }

    calculateHash(index, previousHash, timestamp, data) {
        return CryptoJS.SHA256(index + previousHash + timestamp + JSON.stringify(data)).toString();
    }

    addBlock(newBlock) {
        newBlock.previousHash = this.getLatestBlock().hash;
        newBlock.hash = this.calculateHash(newBlock.index, newBlock.previousHash, newBlock.timestamp, newBlock.data);
        this.chain.push(newBlock);
    }

    getLatestBlock() {
        return this.chain[this.chain.length - 1];
    }
}

const blockchain = new Blockchain();

function addBlock() {
    const latestBlock = blockchain.getLatestBlock();
    const index = latestBlock.index + 1;
    const timestamp = new Date().toISOString();
    const data = `Block ${index} data`;
    const newBlock = new Block(index, latestBlock.hash, timestamp, data, '');
    blockchain.addBlock(newBlock);
    displayBlockchain();
}

function displayBlockchain() {
    const output = document.getElementById('output');
    output.textContent = JSON.stringify(blockchain.chain, null, 4);
}

// Display initial blockchain
displayBlockchain();

