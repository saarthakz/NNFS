import { Matrix, MatrixColumnSelectionView } from "ml-matrix";

class DenseLayer {

  #input: Matrix = new Matrix([]);
  #output: Matrix = new Matrix([]);
  #weights: Matrix;
  #bias: Matrix;
  #inputSize: number;
  #outputSize: number;

  constructor(inputSize: number, outputSize: number) {
    this.#inputSize = inputSize;
    this.#outputSize = outputSize;

    this.#weights = Matrix.random(outputSize, inputSize, {
      random: () => (Math.random() * 2) - 1
    });
    this.#bias = Matrix.zeros(outputSize, 1);
  };

  forward(input: number[]) {

    this.#input = new Matrix([input]).transpose();
    const weightedSum = this.#weights.mmul(this.#input);
    this.#output = weightedSum.add(this.#bias);

    return this.#output.to1DArray();
  };

  backward(outputGradient: number[], learningRate: number) {

    const outputGradientMatrix = new Matrix([outputGradient]).transpose();

    const weightsGradient = outputGradientMatrix.mmul(this.#input.transpose());
    const biasGradient = outputGradientMatrix;
    const inputGradient = this.#weights.transpose().mmul(outputGradientMatrix);

    // console.log("Weights before: ", this.#weights);

    this.#weights.sub(Matrix.mul(weightsGradient, learningRate));
    this.#bias.sub(Matrix.mul(biasGradient, learningRate));

    // console.log("Weights after: ", this.#weights);
    return inputGradient.to1DArray();
  };

};

export default DenseLayer;