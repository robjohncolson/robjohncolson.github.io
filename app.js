// Initialize Supabase
const SUPABASE_URL = '// Initialize Supabase
const SUPABASE_URL = 'https://dlslkmvqjhvsxkzpifpe.supabase.co'; // Replace with your Supabase project URL
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsc2xrbXZxamh2c3hrenBpZnBlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYyNzc3MjksImV4cCI6MjA0MTg1MzcyOX0.l_fMboFPanLTtT1GLnAaDcjLxJ-oJ_duXvOd87kvF3M'; // Replace with your Supabase anon key
const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

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
document.getElementById('request-pass').addEventListener('click', async function() {
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
            time: new Date().toISOString(), // Save in ISO format
            status: 'Pending',  // Status is initially pending
            message: '' // Message field for any notes from the teacher
        };

        // Insert the pass request into the Supabase database
        const { data, error } = await supabase
            .from('pass_requests')  // Make sure the table is named 'pass_requests'
            .insert([request]);

        if (error) {
            console.error('Error submitting pass request:', error.message);
        } else {
            console.log('Pass request submitted:', data);
        }
    }
});

// Listen for all pass requests (teacher view) and update in real-time
async function listenForPassRequests() {
    const { data, error } = await supabase
        .from('pass_requests')
        .select('*');

    if (error) {
        console.error('Error fetching pass requests:', error.message);
        return;
    }

    const passList = document.getElementById('pass-list');
    passList.innerHTML = ''; // Clear the list before adding updates

    data.forEach((request) => {
        const listItem = document.createElement('li');
        listItem.textContent = `${request.name} requested a ${request.pass} pass at ${new Date(request.time).toLocaleTimeString()}`;

        // Approve button for teacher
        const approveButton = document.createElement('button');
        approveButton.textContent = 'Approve';
        approveButton.addEventListener('click', () => approveRequest(request.id));
        listItem.appendChild(approveButton);

        // Deny button for teacher
        const denyButton = document.createElement('button');
        denyButton.textContent = 'Deny';
        denyButton.addEventListener('click', () => denyRequest(request.id));
        listItem.appendChild(denyButton);

        passList.appendChild(listItem);
    });
}

// Listen for student's own pass requests (student view)
async function listenForStudentPassRequests(studentName) {
    const { data, error } = await supabase
        .from('pass_requests')
        .select('*')
        .eq('name', studentName); // Only select the student's requests

    if (error) {
        console.error('Error fetching student pass requests:', error.message);
        return;
    }

    const studentPassList = document.getElementById('student-pass-list');
    studentPassList.innerHTML = ''; // Clear the list before adding updates

    data.forEach((request) => {
        const listItem = document.createElement('li');
        listItem.textContent = `You requested a ${request.pass} pass at
'; // Replace with your Supabase project URL
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY'; // Replace with your Supabase anon key
const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

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
document.getElementById('request-pass').addEventListener('click', async function() {
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
            time: new Date().toISOString(), // Save in ISO format
            status: 'Pending',  // Status is initially pending
            message: '' // Message field for any notes from the teacher
        };

        // Insert the pass request into the Supabase database
        const { data, error } = await supabase
            .from('pass_requests')  // Make sure the table is named 'pass_requests'
            .insert([request]);

        if (error) {
            console.error('Error submitting pass request:', error.message);
        } else {
            console.log('Pass request submitted:', data);
        }
    }
});

// Listen for all pass requests (teacher view) and update in real-time
async function listenForPassRequests() {
    const { data, error } = await supabase
        .from('pass_requests')
        .select('*');

    if (error) {
        console.error('Error fetching pass requests:', error.message);
        return;
    }

    const passList = document.getElementById('pass-list');
    passList.innerHTML = ''; // Clear the list before adding updates

    data.forEach((request) => {
        const listItem = document.createElement('li');
        listItem.textContent = `${request.name} requested a ${request.pass} pass at ${new Date(request.time).toLocaleTimeString()}`;

        // Approve button for teacher
        const approveButton = document.createElement('button');
        approveButton.textContent = 'Approve';
        approveButton.addEventListener('click', () => approveRequest(request.id));
        listItem.appendChild(approveButton);

        // Deny button for teacher
        const denyButton = document.createElement('button');
        denyButton.textContent = 'Deny';
        denyButton.addEventListener('click', () => denyRequest(request.id));
        listItem.appendChild(denyButton);

        passList.appendChild(listItem);
    });
}

// Listen for student's own pass requests (student view)
async function listenForStudentPassRequests(studentName) {
    const { data, error } = await supabase
        .from('pass_requests')
        .select('*')
        .eq('name', studentName); // Only select the student's requests

    if (error) {
        console.error('Error fetching student pass requests:', error.message);
        return;
    }

    const studentPassList = document.getElementById('student-pass-list');
    studentPassList.innerHTML = ''; // Clear the list before adding updates

    data.forEach((request) => {
        const listItem = document.createElement('li');
        listItem.textContent = `You requested a ${request.pass} pass at
