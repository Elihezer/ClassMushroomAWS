function setLanguage(language) {

    if (language === "fr") {

        cameraTitle.innerText =
            "Prendre une photo";

        imageTitle.innerText =
            "Ou choisir une image";

        identifyButton.innerText =
            "Identifier";

    } else {

        cameraTitle.innerText =
            "Take a photo";

        imageTitle.innerText =
            "Or choose an image";

        identifyButton.innerText =
            "Identify";
    }

    localStorage.setItem(
        "language",
        language
    );
}