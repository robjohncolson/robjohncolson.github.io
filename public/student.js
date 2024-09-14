document.getElementById('submit-pass').addEventListener('click', () => {
    const studentName = document.getElementById('student-name').value.trim();
    const passType = document.getElementById('pass-type').value;

    if (studentName && passType) {
        // Send POST request to submit pass request
        fetch('/pass-request', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ studentName, passType })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Pass request submitted:', data);
            loadStudentPassRequests();  // Reload the student's pass requests
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please fill in both your name and pass type.');
    }
});

function loadStudentPassRequests() {
    // Fetch the pass requests and display only the ones for the current student
    fetch('/pass-requests')
        .then(response => response.json())
        .then(data => {
            const studentName = document.getElementById('student-name').value.trim();
            const passList = document.getElementById('student-pass-list');
            passList.innerHTML = '';  // Clear the list

            data.forEach(request => {
                if (request.studentName === studentName) {
                    const listItem = document.createElement('li');
                    listItem.textContent = `${request.passType} - Status: ${request.status}`;
                    passList.appendChild(listItem);
                }
            });
        })
        .catch(error => console.error('Error:', error));
}
