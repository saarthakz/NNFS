import { dReLU, reLU } from "./Activators/relu";
import DenseLayer from "./Layers/DenseLayer";
import { HiddenActivationLayer, SoftmaxOutputActivationLayer } from "./Layers/ActivationLayer";
import { softmax } from "./Activators/softmax";
import imageToArray from "./util/imageToArray";

(async () => {
  const hiddenLayer = new DenseLayer(784, 16);
  const hiddenActivationLayer = new HiddenActivationLayer(reLU, dReLU);
  const outputLayer = new DenseLayer(16, 10);
  const softmaxOutputActivationLayer = new SoftmaxOutputActivationLayer(softmax);

  const imageLabel = 0;
  const learningRate = 0.001;
  const image = await imageToArray("./Data/Training-Data/0/img_0.jpg");
  let oneHotEncoding = new Array<number>(10).fill(0);
  oneHotEncoding[imageLabel] = 1;

  let hiddenOutput = hiddenLayer.forward(image);
  let hiddenActivationOutput = hiddenActivationLayer.forward(hiddenOutput);
  let output = outputLayer.forward(hiddenActivationOutput);
  let softmaxOutput = softmaxOutputActivationLayer.forward(output);
  let loss = -Math.log(softmaxOutput[imageLabel]);
  console.log("Before", loss);

  let outputGradient = softmaxOutputActivationLayer.backward(oneHotEncoding); /* dL/dO*/
  let hiddenActivationOutputGradient = outputLayer.backward(outputGradient, learningRate);
  let hiddenOutputGradient = hiddenActivationLayer.backward(hiddenActivationOutputGradient);
  hiddenLayer.backward(hiddenOutputGradient, learningRate);

  hiddenOutput = hiddenLayer.forward(image);
  hiddenActivationOutput = hiddenActivationLayer.forward(hiddenOutput);
  output = outputLayer.forward(hiddenActivationOutput);
  softmaxOutput = softmaxOutputActivationLayer.forward(output);
  loss = -Math.log(softmaxOutput[imageLabel]);
  console.log("After", loss);

  // console.log(hiddenOutput);

  // const hiddenActivationOutput = hiddenActivationLayer.forward(hiddenOutput);

})();




