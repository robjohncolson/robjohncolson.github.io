let serverDevice;

// Start the Bluetooth server on the teacher's device
document.getElementById('start-bluetooth').addEventListener('click', async () => {
    try {
        serverDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: ['battery_service'] }] // Define your own service UUID
        });

        document.getElementById('status').textContent = 'Connected to student devices...';

        // Handle pass requests from students here
        // For example, you could transmit data or receive student requests
    } catch (error) {
        console.error('Bluetooth error:', error);
        document.getElementById('status').textContent = 'Failed to connect.';
    }
});
