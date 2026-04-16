function setThemeIcon(isDark) {
    const toggleBtn = document.getElementById('theme-toggle');
    if (!toggleBtn) {
        return;
    }
    toggleBtn.textContent = isDark ? '☀️' : '🌙';
    toggleBtn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    setThemeIcon(isDark);
}

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const isDark = savedTheme === 'dark';
    document.body.classList.toggle('dark-mode', isDark);
    setThemeIcon(isDark);

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleTheme);
    }
}

function initializePopovers() {
    if (!window.bootstrap || !bootstrap.Popover) {
        return;
    }
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.forEach((popoverTriggerEl) => {
        bootstrap.Popover.getOrCreateInstance(popoverTriggerEl);
    });
}

function initializeTypingHero() {
    const typingText = document.getElementById('typing-text');
    if (!typingText) {
        return;
    }

    const text = 'Welcome to Datavista';
    let index = 0;
    let isDeleting = false;

    function typeWriter() {
        if (!isDeleting) {
            typingText.textContent = text.slice(0, index + 1);
            index += 1;
            if (index === text.length) {
                isDeleting = true;
                setTimeout(typeWriter, 1600);
                return;
            }
        } else {
            typingText.textContent = text.slice(0, index);
            index -= 1;
            if (index === 0) {
                isDeleting = false;
            }
        }
        setTimeout(typeWriter, isDeleting ? 90 : 130);
    }

    setTimeout(typeWriter, 750);

    typingText.addEventListener('mouseenter', () => typingText.classList.add('glow'));
    typingText.addEventListener('mouseleave', () => typingText.classList.remove('glow'));
    typingText.addEventListener('click', () => {
        typingText.classList.add('pulse');
        setTimeout(() => typingText.classList.remove('pulse'), 450);
    });
}

function initializeCardReveal() {
    const cards = Array.from(document.querySelectorAll('.card-container'));
    if (!cards.length) {
        return;
    }

    cards.forEach((card, idx) => {
        card.classList.add('animate-in');
        card.style.transitionDelay = `${Math.min(idx * 65, 390)}ms`;
    });

    if (!('IntersectionObserver' in window)) {
        cards.forEach((card) => card.classList.add('is-visible'));
        return;
    }

    const observer = new IntersectionObserver(
        (entries, obs) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    obs.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.15 }
    );

    cards.forEach((card) => observer.observe(card));
}

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    initializePopovers();
    initializeTypingHero();
    initializeCardReveal();
});