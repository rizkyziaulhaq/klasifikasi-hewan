document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const imageFile = document.getElementById("imageFile");
  const imagePreview = document.getElementById("imagePreview");
  const previewContainer = document.getElementById("previewContainer");
  const resultContainer = document.getElementById("resultContainer");
  const predictionText = document.getElementById("prediction");
  const confidenceText = document.getElementById("confidence");

  // Menampilkan preview gambar
  imageFile.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
              imagePreview.src = e.target.result;
              imagePreview.style.display = "block";
          };
          reader.readAsDataURL(file);
      }
  });

  // Menangani submit form menggunakan AJAX
  form.addEventListener("submit", (event) => {
      event.preventDefault(); // Mencegah reload halaman

      const formData = new FormData();
      formData.append("file", imageFile.files[0]);

      fetch("/predict", {
          method: "POST",
          body: formData,
      })
          .then((response) => response.json())
          .then((data) => {
              // Menampilkan hasil prediksi
              predictionText.textContent = data.prediction;
              confidenceText.textContent = `${(data.confidence * 100).toFixed(2)}%`;
              resultContainer.style.display = "block";
          })
          .catch((error) => {
              console.error("Error:", error);
              alert("Error while predicting!");
          });
  });
});
