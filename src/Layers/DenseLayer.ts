import { add, matrix, multiply, size, subtract } from "mathjs";
import { Matrix } from "ml-matrix";

class DenseLayer {

  #input: number[] = [];
  #output: number[] = [];
  #weights: number[][];
  #bias: number[];
  #inputSize: number;
  #outputSize: number;

  constructor(inputSize: number, outputSize: number) {
    this.#inputSize = inputSize;
    this.#outputSize = outputSize;

    this.#weights = new Array<number[]>(outputSize)
      .fill([])
      .map(() => new Array<number>(inputSize)
        .fill(0)
        .map(() => Math.random()));

    this.#bias = new Array(outputSize).fill(0);
  };

  forward(input: number[]) {
    this.#input = input;
    return add(multiply(matrix(this.#weights), this.#input), this.#bias);
  };

  backward(outputGradient: number[], learningRate: number) {

    let weightsGradient: number[][] = new Array<number[]>(this.#outputSize)
      .fill([])
      .map(() => new Array<number>(this.#inputSize)
        .fill(0));

    outputGradient.forEach((partialDerivative, idx) => this.#input.forEach((input, _idx) => weightsGradient[idx][_idx] = partialDerivative * input));

    const biasGradient = outputGradient;

    let weightsTranspose: number[][] = new Array<number[]>(this.#inputSize)
      .fill([])
      .map(() => new Array<number>(this.#outputSize)
        .fill(0));

    for (let idx = 0; idx < this.#inputSize; idx++) {
      for (let _idx = 0; _idx < this.#outputSize; _idx++) {
        weightsTranspose[idx][_idx] = this.#weights[_idx][idx];
      };
    };

    const inputGradient = multiply(matrix(weightsTranspose), outputGradient);

  };
};

export default DenseLayer;