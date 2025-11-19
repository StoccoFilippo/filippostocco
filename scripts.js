
// shoe the current time
const now = new Date();
const timeOptions = { hour: '2-digit', minute: '2-digit' };
const timeFormatted = now.toLocaleTimeString('en-US', timeOptions);
document.getElementById('timehr').textContent = timeFormatted;

// Show current date
const dateOptions = { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' };
const dateFormatted = now.toLocaleDateString('en-US', dateOptions);
document.getElementById('datetime').textContent = dateFormatted;

