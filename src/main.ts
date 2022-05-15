import { dReLU, reLU } from "./Activators/relu";
import DenseLayer from "./Layers/DenseLayer";
import { HiddenActivationLayer, SoftmaxOutputActivationLayer } from "./Layers/ActivationLayer";
import { softmax } from "./Activators/softmax";
import { readFileSync } from "fs";
import imageToArray from "./util/imageToArray";
import math from "mathjs";

(async () => {
  const hiddenLayer = new DenseLayer(784, 16);
  const hiddenActivationLayer = new HiddenActivationLayer(reLU, dReLU);
  const outputLayer = new DenseLayer(16, 10);
  const softmaxOutputActivationLayer = new SoftmaxOutputActivationLayer(softmax);
  const image = await imageToArray("./Data/Training-Data/0/img_0.jpg");

  const hiddenOutput = hiddenLayer.forward(image);
  console.log(hiddenOutput);


  // const hiddenActivationOutput = hiddenActivationLayer.forward();

})();



