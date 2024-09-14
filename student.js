let teacherDevice;

// Connect to the teacher's Bluetooth server
document.getElementById('connect-teacher').addEventListener('click', async () => {
    try {
        teacherDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: ['battery_service'] }] // Use your teacher's Bluetooth service UUID
        });

        console.log('Connected to teacher’s device');
    } catch (error) {
        console.error('Bluetooth error:', error);
    }
});

// Send pass request to the teacher
document.getElementById('send-request').addEventListener('click', () => {
    const passRequest = document.getElementById('pass-request').value;

    // Send the pass request to the teacher over Bluetooth
    // For example, use Bluetooth GATT communication to transmit data
    console.log(`Pass request sent: ${passRequest}`);
});
