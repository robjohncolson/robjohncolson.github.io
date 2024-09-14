let orbitdb, db, currentUser;

// Initialize IPFS and OrbitDB
async function initOrbitDB() {
    const ipfs = await Ipfs.create();  // Initialize IPFS
    orbitdb = await OrbitDB.createInstance(ipfs);  // Initialize OrbitDB
    db = await orbitdb.docs('classroom.pass', { indexBy: 'id' });  // Create or open the database
    await db.load();  // Load the database

    // Load all pass requests
    displayPassRequests();
}

// Login logic for teacher and student roles
document.getElementById('login-button').addEventListener('click', () => {
    currentUser = document.getElementById('name').value.trim();

    if (currentUser) {
        document.getElementById('login-section').style.display = 'none';  // Hide login section

        if (currentUser.toLowerCase() === 'teacher') {
            document.getElementById('teacher-view').style.display = 'block';  // Show teacher view
            displayPassRequests();
        } else {
            document.getElementById('student-view').style.display = 'block';  // Show student view
            listenForStudentPassRequests();
        }
    }
});

// Add pass request (Student)
document.getElementById('submit-pass').addEventListener('click', async () => {
    const passType = document.getElementById('pass-type').value;
    let passReason = passType;

    if (passType === 'Other') {
        passReason = document.getElementById('other-reason').value.trim();
    }

    const id = Date.now().toString();  // Unique ID for each pass request
    await db.put({
        id,
        studentName: currentUser,
        passType: passReason,
        status: 'Pending',
        timestamp: new Date().toISOString(),
    });

    listenForStudentPassRequests();
});

// Display all pass requests (Teacher)
async function displayPassRequests() {
    const requests = db.get('');
    const passList = document.getElementById('pass-list');
    passList.innerHTML = '';  // Clear list

    requests.forEach(request => {
        const listItem = document.createElement('li');
        listItem.textContent = `${request.studentName} requested ${request.passType} at ${new Date(request.timestamp).toLocaleTimeString()} - Status: ${request.status}`;

        if (currentUser.toLowerCase() === 'teacher' && request.status === 'Pending') {
            const approveButton = document.createElement('button');
            approveButton.textContent = 'Approve';
            approveButton.addEventListener('click', () => updatePassStatus(request.id, 'Approved'));
            listItem.appendChild(approveButton);

            const denyButton = document.createElement('button');
            denyButton.textContent = 'Deny';
            denyButton.addEventListener('click', () => updatePassStatus(request.id, 'Denied'));
            listItem.appendChild(denyButton);
        }

        passList.appendChild(listItem);
    });
}

// Update pass request status (Teacher)
async function updatePassStatus(id, status) {
    const request = db.get(id)[0];
    request.status = status;
    await db.put(request);
    displayPassRequests();
}

// Listen for student's own pass requests (Student)
function listenForStudentPassRequests() {
    const studentPassList = document.getElementById('student-pass-list');
    const requests = db.get('');

    studentPassList.innerHTML = '';  // Clear list
    requests.forEach(request => {
        if (request.studentName === currentUser) {
            const listItem = document.createElement('li');
            listItem.textContent = `${request.passType} at ${new Date(request.timestamp).toLocaleTimeString()} - Status: ${request.status}`;
            studentPassList.appendChild(listItem);
        }
    });
}

// Initialize the app when the page loads
initOrbitDB();
