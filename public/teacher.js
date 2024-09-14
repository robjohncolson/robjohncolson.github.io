function loadTeacherPassRequests() {
    // Fetch all pass requests
    fetch('/pass-requests')
        .then(response => response.json())
        .then(data => {
            const passList = document.getElementById('teacher-pass-list');
            passList.innerHTML = '';  // Clear the list

            data.forEach(request => {
                const listItem = document.createElement('li');
                listItem.textContent = `${request.studentName} requested ${request.passType} - Status: ${request.status}`;

                if (request.status === 'Pending') {
                    const approveButton = document.createElement('button');
                    approveButton.textContent = 'Approve';
                    approveButton.addEventListener('click', () => updateRequestStatus(request.id, 'Approved'));
                    listItem.appendChild(approveButton);

                    const denyButton = document.createElement('button');
                    denyButton.textContent = 'Deny';
                    denyButton.addEventListener('click', () => updateRequestStatus(request.id, 'Denied'));
                    listItem.appendChild(denyButton);
                }

                passList.appendChild(listItem);
            });
        })
        .catch(error => console.error('Error:', error));
}

function updateRequestStatus(requestId, status) {
    fetch(`/pass-request/${requestId}`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ status })
    })
    .then(response => response.json())
    .then(data => {
        console.log(`Pass request ${status.toLowerCase()}:`, data);
        loadTeacherPassRequests();  // Reload the teacher's pass requests
    })
    .catch(error => console.error('Error:', error));
}

// Load pass requests when the page loads
loadTeacherPassRequests();
