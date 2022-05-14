import { dReLU, reLU } from "./Activators/relu";
import DenseLayer from "./Layers/DenseLayer";
import { HiddenActivationLayer, SoftmaxOutputActivationLayer } from "./Layers/ActivationLayer";
import { softmax } from "./Activators/softmax";

const hiddenLayer = new DenseLayer(784, 16);
const hiddenActivationLayer = new HiddenActivationLayer(reLU, dReLU);
const outputLayer = new DenseLayer(16, 10);
const softmaxOutputActivationLayer = new SoftmaxOutputActivationLayer(softmax);
