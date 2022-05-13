import { dReLU, reLU } from "./Activators/relu";
import DenseLayer from "./Layers/DenseLayer";
import { HiddenActivationLayer } from "./Layers/ActivationLayer";

const hiddenLayer = new DenseLayer(784, 16);
const hiddenActivationLayer = new HiddenActivationLayer(reLU, dReLU);
const outputLayer = new DenseLayer(16, 10);




