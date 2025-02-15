document.addEventListener("DOMContentLoaded", function () {
    console.log("JavaScript Loaded!");

    // Function to upload image & process it
    async function uploadImage(inputId, apiEndpoint, resultImgId) {
        const input = document.getElementById(inputId);
        const file = input.files[0];

        if (!file) {
            alert("Please select an image file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // Show loading message
        const resultImg = document.getElementById(resultImgId);
        resultImg.style.display = "none"; // Hide previous result
        resultImg.alt = "Processing...";
        
        try {
            const response = await fetch(`http://127.0.0.1:8000/${apiEndpoint}/`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server Error: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.filename) {
                const imageUrl = `http://127.0.0.1:8000/results/${data.filename}`;
                resultImg.src = imageUrl;
                resultImg.style.display = "block";
                resultImg.alt = "Processed Image";
            } else {
                alert("Processing failed. Try again.");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while processing the image.");
        }
    }

    // Attach event listeners to buttons
    document.getElementById("cloudUploadBtn").addEventListener("click", function () {
        uploadImage("cloudUpload", "remove-clouds", "cloudResult");
    });

    document.getElementById("segmentUploadBtn").addEventListener("click", function () {
        uploadImage("segmentUpload", "analyze-image", "segmentResult");
    });
});