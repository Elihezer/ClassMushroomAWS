HTML_PAGE = """
<!DOCTYPE html>

<html lang="fr">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>Mushroom AI</title>

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


<h3>Prendre une photo</h3>

<input
    id="cameraInput"
    type="file"
    accept="image/*"
    capture="environment"
>


<h3>Ou choisir une image</h3>

<input
    id="imageInput"
    type="file"
    accept="image/*"
>


<img id="preview">


<button id="identifyButton">
    Identifier
</button>


<div id="status"></div>

<div id="results"></div>


<p>
    Experimental classifier.
    Never use the result to decide
    whether a mushroom is safe to eat.
</p>


<script>

let selectedFile = null;


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

    preview.src =
        URL.createObjectURL(
            file
        );

    preview.style.display =
        "block";
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


function prepareImage(file) {

    return new Promise(
        function (
            resolve,
            reject
        ) {

            const reader =
                new FileReader();

            reader.onload =
                function (event) {

                    const image =
                        new Image();


                    image.onload =
                        function () {

                            const maxSize =
                                1280;


                            let width =
                                image.width;

                            let height =
                                image.height;


                            if (
                                width > height
                                &&
                                width > maxSize
                            ) {

                                height =
                                    height
                                    * maxSize
                                    / width;

                                width =
                                    maxSize;

                            }


                            if (
                                height >= width
                                &&
                                height > maxSize
                            ) {

                                width =
                                    width
                                    * maxSize
                                    / height;

                                height =
                                    maxSize;

                            }


                            const canvas =
                                document.createElement(
                                    "canvas"
                                );


                            canvas.width =
                                Math.round(
                                    width
                                );

                            canvas.height =
                                Math.round(
                                    height
                                );


                            const context =
                                canvas.getContext(
                                    "2d"
                                );


                            context.drawImage(
                                image,
                                0,
                                0,
                                canvas.width,
                                canvas.height
                            );


                            const dataUrl =
                                canvas.toDataURL(
                                    "image/jpeg",
                                    0.85
                                );


                            const base64Image =
                                dataUrl.split(
                                    ","
                                )[1];


                            resolve(
                                base64Image
                            );

                        };


                    image.onerror =
                        reject;


                    image.src =
                        event.target.result;

                };


            reader.onerror =
                reject;


            reader.readAsDataURL(
                file
            );

        }
    );

}


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
                "Choisis une image.";

            return;

        }


        status.innerText =
            "Analyse...";


        results.innerHTML =
            "";


        try {

            const imageBase64 =
                await prepareImage(
                    selectedFile
                );


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