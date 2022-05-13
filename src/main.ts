import DenseLayer from "./Layers/DenseLayer";

const dense = new DenseLayer(10, 4);

let output = dense.forward(Array(10).fill(1));

dense.backward([1, 2, 3, 4], 0.1);
