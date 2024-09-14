// Use the JS version of IPFS
const IPFS = require('ipfs');
const OrbitDB = require('orbit-db');

async function start() {
  const ipfs = await IPFS.create();
  const orbitdb = await OrbitDB.createInstance(ipfs);

  // Open or create the classroom pass database
  const db = await orbitdb.docs('classroom.pass', { indexBy: 'id' });
  await db.load();

  // Function to add a pass request
  async function addPassRequest() {
    const id = Date.now().toString();
    await db.put({
      id,
      studentName: 'Jane Doe',
      passType: 'Bathroom',
      status: 'Pending',
      timestamp: new Date().toISOString(),
    });
    displayPassRequests();
  }

  // Function to display all pass requests
  function displayPassRequests() {
    const requests = db.get('');
    const requestDiv = document.getElementById('pass-requests');
    requestDiv.innerHTML = JSON.stringify(requests, null, 2);
  }

  document.getElementById('submit-pass').addEventListener('click', addPassRequest);
  displayPassRequests();
}

start();
