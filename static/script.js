const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

sign_up_btn.addEventListener('click',()=>{
  container.classList.add("sign-up-mode");

});

sign_in_btn.addEventListener('click',()=>{
  container.classList.remove("sign-up-mode");

});

document.querySelectorAll('.fa-eye').forEach(eyeIcon => {
    eyeIcon.addEventListener('click', (event) => {
        event.preventDefault(); // Prevent default click behavior
        const passwordField = eyeIcon.previousElementSibling;
        if (passwordField.type === 'password') {
            passwordField.type = 'text';
            eyeIcon.classList.remove('fa-eye');
            eyeIcon.classList.add('fa-eye-slash');
        } else {
            passwordField.type = 'password';
            eyeIcon.classList.remove('fa-eye-slash');
            eyeIcon.classList.add('fa-eye');
        }
    });

   
});
function checkPassword() {
    let password = document.getElementById("password").value;
    let cnfrmPassword = document.getElementById("cnfrm-password").value;
    console.log(" Password:", password, '\n', "Confirm Password:", cnfrmPassword);
    let message = document.getElementById("message");

    if (cnfrmPassword.length != 0) {
        if (password == cnfrmPassword) {
            message.textContent = "Le mot de passe correspond";
            message.className = "message success";
            message.style.display = "block";
        } else {
            message.textContent = "Le mot de passe ne correspond pas";
            message.className = "message error";
            message.style.display = "block";
        }
    } else {
        message.textContent = "";
        message.style.display = "none";
    }
}




function validatePassword() {
    const passwordInput = document.getElementById('password');
    const password = passwordInput.value;
    const messagee = document.getElementById('messagee');
    const submitBtn = document.getElementById('submitBtn');

    const hasNumber = /\d/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasUpperCase = /[A-Z]/.test(password);
    const hasSpecialChar = /[!@#$%^&*]/.test(password);
    const isLengthValid = password.length >= 8;

    if (hasNumber && hasLowerCase && hasUpperCase && hasSpecialChar && isLengthValid) {
        submitBtn.disabled = false;
        messagee.textContent = '';
    } else {
        submitBtn.disabled = true;
        messagee.textContent = 'Le mot de passe doit contenir au moins 8 caractères, une minuscule, une majuscule, un chiffre et un caractère spécial.';
        messagee.color = red;
    }
}

const passwordinput = document.querySelector(".pass-field input");
const content = document.querySelector(".conteent");

const requirements = [
    { regex: /.{8,}/, index: 0 },
    { regex: /.*[0-9].*/, index: 1 },
    { regex: /.*[a-z].*/, index: 2 },
    { regex: /.*[^A-Za-z0-9].*/, index: 3 },
    { regex: /.*[A-Z].*/, index: 4 }
];

passwordinput.addEventListener("keyup", () => {
    let allRequirementsMet = true;

    requirements.forEach(item => {
        const isValid = item.regex.test(passwordinput.value);
        const requirementItem = content.querySelector('.requirement-list li:nth-child(' + (item.index + 1) + ')');

        if (isValid) {
            requirementItem.firstElementChild.className = "fa-solid fa-check";
            requirementItem.classList.add("valid");
        } else {
            requirementItem.firstElementChild.className = "fa-solid fa-circle";
            allRequirementsMet = false;
            requirementItem.classList.remove("valid");
        }
    });

    if (allRequirementsMet) {
        content.style.display = "none";
    } else {
        content.style.display = "block";
    }
});



