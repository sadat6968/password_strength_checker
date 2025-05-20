function checkStrength() {
    let pass = document.getElementById("password").value;
    let strengthText = document.getElementById("strength");
    let strength = "Weak";
    let className = "weak";

    if (pass.length > 8 && /\d/.test(pass) && /[A-Z]/.test(pass) && /[^a-zA-Z\d]/.test(pass)) {
        strength = "Strong";
        className = "strong";
    } else if (pass.length >= 6) {
        strength = "Moderate";
        className = "moderate";
    }

    strengthText.innerText = "Strength: " + strength;
    strengthText.className = className;
}
