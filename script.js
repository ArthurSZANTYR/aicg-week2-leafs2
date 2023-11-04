let model; //var globale pour le model

var loadFile = function(event) {  //run quand il y a un fichier selectionné

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

            canvas.width = 200 ;  //attention c'est la taille du canva pas de l'image
            canvas.height = 200 ;

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height );  //draw image
            image_view.appendChild(canvas); // Display editied image in canvas  //image_view cet le nom du divider html
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height) // get image data from the canva
            

            imageTensor = transform_image(imageData);
            console.log(imageTensor.dims)

            const prediction = await predict(imageTensor)
            console.log(prediction)
        }
    })
};

function transform_image(img){
    const inputArray = new Float32Array(1 * 3 * 200 * 200);//création d'un array pour stocker les données vidéo

    // Copy image data into the inputArray
    for (let i = 0; i < 200 * 200; i++) {
      inputArray[i * 3] = img.data[i * 4] / 255;    //R
      inputArray[i * 3 + 1] = img.data[i * 4 + 1] / 255; //G
      inputArray[i * 3 + 2] = img.data[i * 4 + 2] / 255;  //B
    }

    const tensor_image = new ort.Tensor('float32', inputArray, [1, 3, 200, 200]);  //converti le array en tensor

    return tensor_image
}

async function predict(transformedImage) {

    const ortOutputs = await model.run({ 'input': transformedImage }); //on obtient le tenseur de proba
  
    const outputData = ortOutputs.output.data; 
    const probas = softmax(outputData);

    const predictedClassIndex = findMaxIndex(probas);

    const classLabels = ['healthy', 'multiple_diseases', 'rust', 'scab'];
    const predictedClassName = classLabels[predictedClassIndex];

    return predictedClassName;
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
  



