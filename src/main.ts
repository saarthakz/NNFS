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

  const image = await imageToArray("./Data/Training-Data/0/img_0.jpg");

  const hiddenOutput = hiddenLayer.forward(image);
  const hiddenActivationOutput = hiddenActivationLayer.forward(hiddenOutput);
  const output = outputLayer.forward(hiddenActivationOutput);
  const softmaxOutput = softmaxOutputActivationLayer.forward(output);
  console.log(softmaxOutput);


  let oneHotEncoding = new Array<number>(10).fill(0);
  oneHotEncoding[0] = 1;
  // console.log(hiddenOutput);

  // const hiddenActivationOutput = hiddenActivationLayer.forward(hiddenOutput);

})();



