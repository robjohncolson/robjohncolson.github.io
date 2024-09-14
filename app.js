let blockchain = new Blockchain();  // Initialize blockchain for storing pass requests
let currentUser = '';  // Store the name of the current user

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
            displayStudentPassRequests();
        }
    } else {
        alert('Please enter your name to continue.');
    }
});

// Function to submit a pass request (Student)
document.getElementById('submit-pass').addEventListener('click', () => {
    const passType = document.getElementById('pass-type').value;
    if (!passType) {
        alert('Please select a pass type.');
        return;
    }
    const newRequest = {
        studentName: currentUser,
        passType: passType,
        status: 'Pending',
        timestamp: new Date().toISOString(),
    };
    blockchain.addBlock(new Block(blockchain.chain.length, new Date(), newRequest));
    displayStudentPassRequests();
});

// Function for the teacher to approve or deny pass requests
function updatePassRequestStatus(blockIndex, status) {
    let block = blockchain.chain[blockIndex];
    if (block && block.data.status === 'Pending') {
        block.data.status = status;
        block.hash = block.calculateHash();
        displayPassRequests();
    } else {
        alert('Invalid or already processed request.');
    }
}

// Display all pass requests (Teacher)
function displayPassRequests() {
    const passList = document.getElementById('pass-list');
    passList.innerHTML = '';  // Clear list

    blockchain.chain.forEach((block, index) => {
        if (index !== 0) {  // Skip the genesis block
            const listItem = document.createElement('li');
            listItem.textContent = `${block.data.studentName} requested ${block.data.passType} at ${new Date(block.data.timestamp).toLocaleTimeString()} - Status: ${block.data.status}`;

            if (block.data.status === 'Pending') {
                const approveButton = document.createElement('button');
                approveButton.textContent = 'Approve';
                approveButton.addEventListener('click', () => updatePassRequestStatus(index, 'Approved'));
                listItem.appendChild(approveButton);

                const denyButton = document.createElement('button');
                denyButton.textContent = 'Deny';
                denyButton.addEventListener('click', () => updatePassRequestStatus(index, 'Denied'));
                listItem.appendChild(denyButton);
            }

            passList.appendChild(listItem);
        }
    });
}

// Display student pass requests (Student)
function displayStudentPassRequests() {
    const studentPassList = document.getElementById('student-pass-list');
    studentPassList.innerHTML = '';  // Clear list

    blockchain.chain.forEach(block => {
        if (block.data.studentName === currentUser) {
            const listItem = document.createElement('li');
            listItem.textContent = `${block.data.passType} at ${new Date(block.data.timestamp).toLocaleTimeString()} - Status: ${block.data.status}`;
            studentPassList.appendChild(listItem);
        }
    });
}
