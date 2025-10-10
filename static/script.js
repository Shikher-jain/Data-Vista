// Theme Toggle Functionality
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    const toggleBtn = document.getElementById('theme-toggle');
    toggleBtn.textContent = isDark ? '☀️' : '🌙';
}

// Load saved theme
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark';
    if (isDark) {
        document.body.classList.add('dark-mode');
    }

    // Set initial button icon
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.textContent = isDark ? '☀️' : '🌙';
        toggleBtn.addEventListener('click', toggleTheme);
    }

    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });



    // Typing animation for hero title
    const typingText = document.getElementById('typing-text');
    const text = "Welcome to Datavista";
    let index = 0;
    let isDeleting = false;

    function typeWriter() {
        if (!isDeleting) {
            typingText.innerHTML = text.slice(0, index + 1);
            index++;
            if (index === text.length) {
                isDeleting = true;
                setTimeout(typeWriter, 2000); // Pause at end
                return;
            }
        } else {
            typingText.innerHTML = text.slice(0, index);
            index--;
            if (index === 0) {
                isDeleting = false;
            }
        }
        setTimeout(typeWriter, isDeleting ? 100 : 150);
    }

    // Start typing animation after a delay
    setTimeout(typeWriter, 1000);

    // Interactive welcome text
    const welcomeText = document.getElementById('typing-text');
    welcomeText.addEventListener('mouseenter', () => {
        welcomeText.classList.add('glow');
    });
    welcomeText.addEventListener('mouseleave', () => {
        welcomeText.classList.remove('glow');
    });
    welcomeText.addEventListener('click', () => {
        welcomeText.classList.add('pulse');
        setTimeout(() => {
            welcomeText.classList.remove('pulse');
        }, 500);
    });
});