HTML_PAGE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>


<title>Mushroom AI</title>

<link
    rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.css"
>

<style>

body {
    font-family: Arial, sans-serif;
    max-width: 600px;
    margin: auto;
    padding: 20px;
}

button {
    width: 100%;
    padding: 15px;
    margin-top: 15px;
    font-size: 18px;
}

input {
    width: 100%;
    margin-top: 10px;
}

img {
    width: 100%;
    margin-top: 15px;
    display: none;
}

.result {
    padding: 10px;
    margin-top: 10px;
    border: 1px solid #aaa;
}

</style>

</head>

<body>

<h1>Mushroom AI</h1>


<h3>Take a picture</h3>

<input
    id="cameraInput"
    type="file"
    accept="image/*"
    capture="environment"
>


<h3>Or choose an image</h3>

<input
    id="imageInput"
    type="file"
    accept="image/*"
>


<img id="preview">


<button id="identifyButton">
    Identify
</button>


<div id="status"></div>

<div id="results"></div>


<p>
    Experimental classifier.
    Never use the result to decide
    whether a mushroom is safe to eat.
</p>

<script
    src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.6.2/cropper.min.js">
</script>

<script>

let selectedFile = null;
let cropper = null;

const cameraInput =
    document.getElementById(
        "cameraInput"
    );

const imageInput =
    document.getElementById(
        "imageInput"
    );

const preview =
    document.getElementById(
        "preview"
    );

const status =
    document.getElementById(
        "status"
    );

const results =
    document.getElementById(
        "results"
    );


function selectFile(file) {


    selectedFile = file;

    if (cropper !== null) {
        cropper.destroy();
        cropper = null;
    }

    const imageUrl =
        URL.createObjectURL(file);

    preview.src = imageUrl;

    preview.style.display =
        "block";

    preview.onload = function () {

        cropper = new Cropper(
            preview,
            {
                viewMode: 1,
                autoCropArea: 0.8,
                responsive: true,
                background: false
            }
        );
    };
}


cameraInput.addEventListener(
    "change",
    function () {

        if (
            cameraInput.files.length
            > 0
        ) {

            selectFile(
                cameraInput.files[0]
            );

        }

    }
);


imageInput.addEventListener(
    "change",
    function () {

        if (
            imageInput.files.length
            > 0
        ) {

            selectFile(
                imageInput.files[0]
            );

        }

    }
);




document
.getElementById(
    "identifyButton"
)
.addEventListener(
    "click",
    async function () {

        if (
            selectedFile === null
        ) {

            status.innerText =
                "Choose an image.";

            return;

        }

        if (
            cropper === null
        ) {

            status.innerText =
                "Image not ready.";

            return;
        }

        
        

        status.innerText =
            "Analyse...";


        results.innerHTML =
            "";


        try {

            const canvas =
                cropper.getCroppedCanvas(
                    {
                        maxWidth: 1280,
                        maxHeight: 1280
                    }
             );

            const dataUrl =
                canvas.toDataURL(
                    "image/jpeg",
                    0.85
                );

            const imageBase64 =
                dataUrl.split(",")[1];


            const response =
                await fetch(
                    window.location.href,
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                {
                                    image:
                                        imageBase64
                                }
                            )

                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
            ) {

                throw new Error(
                    data.error
                );

            }


            status.innerText =
                "";


            data.predictions.forEach(
                function (
                    prediction,
                    index
                ) {

                    const result =
                        document.createElement(
                            "div"
                        );


                    result.className =
                        "result";


                    result.innerText =
                        (index + 1)
                        + ". "
                        + prediction.name
                        + " - "
                        + prediction.probability
                        + "%";


                    results.appendChild(
                        result
                    );

                }
            );


        } catch (error) {

            status.innerText =
                "Erreur : "
                + error.message;

        }

    }
);

</script>

</body>

</html>
"""