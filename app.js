// Initialize Gun (default: works peer-to-peer in browser)
const gun = Gun();

// Gun database for pass requests
const passRequests = gun.get('pass-requests');

// Simple login logic
let currentUser = localStorage.getItem('currentUser'); // Check if user is logged in

// Automatically log back in if the user is stored in localStorage
if (currentUser) {
    if (currentUser.toLowerCase() === 'teacher') {
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('teacher-view').style.display = 'block';
        listenForPassRequests(); // Automatically update for teacher
    } else {
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('pass-form').style.display = 'block';
        listenForStudentPassRequests(currentUser); // Show student's own requests
    }
}

// Handle login
document.getElementById('login-button').addEventListener('click', function() {
    const name = document.getElementById('name').value.trim();
    
    if (name) {
        currentUser = name.toLowerCase();
        localStorage.setItem('currentUser', currentUser); // Store the login state
        document.getElementById('login-form').style.display = 'none';
        
        if (currentUser === 'teacher') {
            document.getElementById('teacher-view').style.display = 'block';
            listenForPassRequests(); // Automatically update for teacher
        } else {
            document.getElementById('pass-form').style.display = 'block';
            listenForStudentPassRequests(currentUser); // Show student's own requests
        }
    }
});

// Show or hide the "Other" text box based on selection
document.getElementById('pass-type').addEventListener('change', function() {
    const otherPass = document.getElementById('other-pass');
    if (this.value === 'Other') {
        otherPass.style.display = 'block';
    } else {
        otherPass.style.display = 'none';
    }
});

// Submit pass request (students)
document.getElementById('request-pass').addEventListener('click', function() {
    const passType = document.getElementById('pass-type').value;
    let passReason = passType;
    
    // If "Other" is selected, use the custom reason
    if (passType === 'Other') {
        passReason = document.getElementById('other-reason').value.trim();
    }

    if (passReason) {
        const request = {
            name: currentUser,
            pass: passReason,
            time: new Date().toLocaleTimeString(),
            student: currentUser,
            status: 'Pending',  // Status is initially pending
            message: '' // Message field for any notes from the teacher
        };

        // Store the pass request in Gun.js database
        passRequests.set(request);
    }
});

// Listen for all pass requests (teacher view) and update in real-time
function listenForPassRequests() {
    passRequests.map().on((request, id) => {
        const passList = document.getElementById('pass-list');
        const existingItem = document.getElementById(id);

        // If the request is deleted (approved/retracted/denied), remove it from the list
        if (!request) {
            if (existingItem) {
                existingItem.remove();
            }
            return;
        }

        // If the request is new, add it to the list
        if (!existingItem) {
            const listItem = document.createElement('li');
            listItem.id = id;  // Set unique ID for the list item
            listItem.textContent = `${request.name} requested a ${request.pass} pass at ${request.time}`;

            // Approve button for teacher
            const approveButton = document.createElement('button');
            approveButton.textContent = 'Approve';
            approveButton.addEventListener('click', () => approveRequest(id));
            listItem.appendChild(approveButton);

            // Deny button for teacher (with a message prompt)
            const denyButton = document.createElement('button');
            denyButton.textContent = 'Deny';
            denyButton.addEventListener('click', () => denyRequest(id));
            listItem.appendChild(denyButton);

            passList.appendChild(listItem);
        }
    });
}

// Listen for only student's own pass requests and update in real-time
function listenForStudentPassRequests(studentName) {
    passRequests.map().on((request, id) => {
        if (request && request.student === studentName) {
            const studentPassList = document.getElementById('student-pass-list');
            const existingItem = document.getElementById(id);

            // If the request is deleted, notify the student that the pass was approved/denied
            if (!request && existingItem) {
                existingItem.remove();
                return;
            }

            // If the request is new or updated, add/update it in the list
            if (!existingItem) {
                const listItem = document.createElement('li');
                listItem.id = id;  // Set unique ID for the list item
                listItem.textContent = `You requested a ${request.pass} pass at ${request.time}. Status: ${request.status}`;
                
                if (request.message) {
                    listItem.textContent += ` - Message from teacher: ${request.message}`;
                }

                // Retract button for students
                const retractButton = document.createElement('button');
                retractButton.textContent = 'Retract';
                retractButton.addEventListener('click', () => retractRequest(id));
                listItem.appendChild(retractButton);

                studentPassList.appendChild(listItem);
            } else {
                // Update the status and message if the request already exists
                existingItem.textContent = `You requested a ${request.pass} pass at ${request.time}. Status: ${request.status}`;
                
                if (request.message) {
                    existingItem.textContent += ` - Message from teacher: ${request.message}`;
                }
            }
        }
    });
}

// Approve pass request (teacher)
function approveRequest(id) {
    passRequests.get(id).put({ status: 'Approved', message: '' }); // Update status to "Approved"
    setTimeout(() => {
        passRequests.get(id).put(null); // Remove the request after a delay
    }, 1000); // Simulate delay to give students time to see "Approved" status
}

// Deny pass request (teacher with custom message)
function denyRequest(id) {
    const message = prompt("Enter a message for the student (optional):");
    passRequests.get(id).put({ status: 'Denied', message: message || 'Pass denied' }); // Update status to "Denied"
    setTimeout(() => {
        passRequests.get(id).put(null); // Remove the request after a delay
    }, 3000); // Allow the student time to see the denial message
}

// Retract pass request (student)
function retractRequest(id) {
    passRequests.get(id).put(null); // Remove the retracted request
}

// Clear login state when needed (optional feature if you want to implement a logout button)
// localStorage.removeItem('currentUser');
