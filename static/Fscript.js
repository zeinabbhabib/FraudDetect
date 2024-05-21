const form = document.querySelector("form"),
fileInput = form.querySelector(".file-input"),
progressArea = document.querySelector(".progress-area"),
uploadedArea = document.querySelector(".uploaded-area");

// Initially hide the progress and uploaded areas
progressArea.style.display = "none";
uploadedArea.style.display = "none";

form.addEventListener("click",() => {
  fileInput.click();
});

fileInput.onchange = ({target}) => {
  let file =target.files[0];
  if (file){
    let fileName = file.name;
    uploadFile(file);
    // Show the progress area once a file is selected
    progressArea.style.display = "block";
  }
}

function uploadFile(file){
  let formData = new FormData();
  formData.append("file", file);

  let xhr = new XMLHttpRequest();
  xhr.open("POST", "/upload");

  xhr.upload.onprogress = function(event) {
    let percent = Math.round((event.loaded / event.total) * 100);
    let progressElement = document.querySelector(".progress");
    let percentElement = document.querySelector(".percent");

    progressElement.style.width = percent + "%";
    percentElement.innerText = percent + "%";

    let nameElement = document.querySelector(".name");
    nameElement.innerText = file.name + " • Téléchargement";
  };

  xhr.onload = function() {
    if (xhr.status == 200) {
      // Show the uploaded area and hide the progress area
      progressArea.style.display = "none";
      uploadedArea.style.display = "block";

      let uploadedElement = document.querySelector(".uploaded-area");
      uploadedElement.innerHTML = `
        <li class="row">
          <div class="content">
            <i class="fas fa-file-alt"></i>
            <div class="details">
              <span class="name">${file.name} • Téléchargement fini</span>
              <span class="size">${(file.size / 1024).toFixed(2)} KB</span>
            </div>
          </div>
          <i class="fas fa-check"></i>
        </li>
      `;
    }
  };

  xhr.send(formData);
}
