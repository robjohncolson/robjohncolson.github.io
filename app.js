// Firebase configuration (from your Firebase project setup)
// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-analytics.js";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBogRWfXMcTcNJ-HomATjBPAtGOQC8hzKY",
  authDomain: "passmanagementsystem.firebaseapp.com",
  projectId: "passmanagementsystem",
  storageBucket: "passmanagementsystem.appspot.com",
  messagingSenderId: "1005886377912",
  appId: "1:1005886377912:web:61cb2ae14852fa3106094d",
  measurementId: "G-HXGNV64DFP"
};


// Initialize Firebase

firebase.initializeApp(firebaseConfig);
const db = firebase.database();

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

        // Store the pass request in Firebase
        db.ref('passRequests').push(request);
    }
});

// Listen for all pass requests (teacher view) and update in real-time
function listenForPassRequests() {
    db.ref('passRequests').on('value', (snapshot) => {
        const passList = document.getElementById('pass-list');
        passList.innerHTML = ''; // Clear the list before adding updates
        snapshot.forEach((childSnapshot) => {
            const request = childSnapshot.val();
            const id = childSnapshot.key;

            const listItem = document.createElement('li');
            listItem.id = id;  // Set unique ID for the list item
            listItem.textContent = `${request.name} requested a ${request.pass} pass at ${request.time}`;

            // Approve button for teacher
            const approveButton = document.createElement('button');
            approveButton.textContent = 'Approve';
            approveButton.addEventListener('click', () => approveRequest(id));
            listItem.appendChild(approveButton);

            // Deny button for teacher
            const denyButton = document.createElement('button');
            denyButton.textContent = 'Deny';
            denyButton.addEventListener('click', () => denyRequest(id));
            listItem.appendChild(denyButton);

            passList.appendChild(listItem);
        });
    });
}

// Listen for only student's own pass requests and update in real-time
function listenForStudentPassRequests(studentName) {
    db.ref('passRequests').on('value', (snapshot) => {
        const studentPassList = document.getElementById('student-pass-list');
        studentPassList.innerHTML = ''; // Clear the list before adding updates
        snapshot.forEach((childSnapshot) => {
            const request = childSnapshot.val();
            const id = childSnapshot.key;

            if (request.student === studentName) {
                const listItem = document.createElement('li');
                listItem.id = id;  // Set unique ID for the list item
                listItem.textContent = `You requested a ${request.pass} pass at ${request.time}. Status
