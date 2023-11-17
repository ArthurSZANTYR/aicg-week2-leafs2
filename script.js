let model; //var globale pour le model

var loadFile = function(event) {  //run quand il y a un fichier selectionné

    clearPreviousResults(); // supprimer les éléments existants

    const fileInput = document.getElementById('input');
    const image_view = document.getElementById('image_view');

    const fr = new FileReader(); 
    fr.readAsDataURL(fileInput.files[0]);

    fr.addEventListener('load', async() => {// This function runs when reading is complete
        
        model = await ort.InferenceSession.create('./onnx_model.onnx');

        const url = fr.result;  
        const img = new Image();
        img.src = url;  //on set l'url de l'image a notr input

        img.onload = async() => {// This function runs when image has loaded

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d'); //on crée un canva 2D

            img.width = 200
            img.height = 200

            canvas.width = img.width;  //attention c'est la taille du canva pas de l'image
            canvas.height = img.height ;

            ctx.drawImage(img, 0, 0, img.width, img.height );  //draw image
            image_view.appendChild(canvas); // Display editied image in canvas  //image_view cet le nom du divider html
            
            const imageData = ctx.getImageData(0, 0, img.width, img.height) // get image data from the canva
            imageTensor = transform_image(imageData);

            const prediction = await predict(imageTensor)
            const classPrediction = prediction[0]
            const probabilityPrediction = prediction[1]

            const output_div = document.getElementById("output_div");
            output_div.innerHTML = "Plants's state : " + classPrediction + "<br>" + 
                       "Confidence : " + probabilityPrediction;
            //output_div.insertAdjacentText("afterend", prediction);

            
        }
    })
};

function transform_image(img){
    const redArray = [];
    const greenArray = [];
    const blueArray = [];

    for (let i = 0; i < 4*200*200; i += 4) {
        redArray.push((img.data[i] - 0.485*255) / (0.229*255));  //on normalize en meme temps 
        greenArray.push((img.data[i + 1] - 0.456*255) / (0.224*255));
        blueArray.push((img.data[i + 2] - 0.406*255) / (0.225*255));
        // Ignore data[i + 3] to filter out the alpha channel
    }

    const transposedData = redArray.concat(greenArray, blueArray);

    let i, l = transposedData.length; // length, we need this for the loop

    const float32Data = new Float32Array(3 * img.width * img.height);

    for (i = 0; i < l; i++) {
        float32Data[i] = transposedData[i]; // convert to float
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, img.width, img.height]);

    return inputTensor;

}



async function predict(transformedImage) {

    const ortOutputs = await model.run({ 'input': transformedImage }); //on obtient le tenseur de proba
  
    const outputData = ortOutputs.output.data; 
    const probas = softmax(outputData);

    const predictedClassIndex = findMaxIndex(probas);
    const classProbability = probas[predictedClassIndex]
    const classLabels = ['healthy', 'multiple_diseases', 'rust', 'scab'];
    const predictedClassName = classLabels[predictedClassIndex];

    return [predictedClassName, classProbability];
  }

function softmax(data) {
  const exps = data.map((value) => Math.exp(value));
  const sumExps = exps.reduce((acc, val) => acc + val);
  return exps.map((exp) => exp / sumExps);
}

function findMaxIndex(arr) {
  let max = arr[0];
  let maxIndex = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function clearPreviousResults() {
    const outputDiv = document.getElementById('output_div');
    const imageView = document.getElementById('image_view');
  
    if (outputDiv) {
      outputDiv.innerHTML = ''; // Vide le contenu de l'élément
    }
    if (imageView) {
      imageView.innerHTML = ''; // Vide le contenu de l'élément
    }
  }


  



