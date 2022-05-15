import { Matrix } from "ml-matrix";

export class HiddenActivationLayer {

  #input: Matrix = new Matrix([]);
  #output: Matrix = new Matrix([]);
  #func;
  #derivative;

  constructor(func: (input: number[]) => number[], derivative: (output: number[]) => number[]) {
    this.#func = func;
    this.#derivative = derivative;
  };

  forward(input: number[]) {
    this.#input = new Matrix([input]).transpose();
    this.#output = new Matrix([this.#func(input)]).transpose();
    return this.#output.to1DArray();
  };

  backward(outputGradient: number[]) {
    const outputGradientMatrix = new Matrix([outputGradient]).transpose();
    const derivatives = this.#derivative(this.#output.to1DArray());
    const inputGradient = outputGradientMatrix.mul(
      new Matrix([derivatives]).transpose()
    );
    return inputGradient.to1DArray();
  };
};

export class SoftmaxOutputActivationLayer {

  #input: number[] = [];
  #output: number[] = [];
  #func;

  constructor(func: (input: number[]) => number[]) {
    this.#func = func;
  };

  forward(input: number[]) {
    this.#input = input;
    this.#output = this.#func(input);
    return this.#output;
  };

  backward(oneHotEncoding: number[]) {
    return this.#output.map((val, idx) => val - oneHotEncoding[idx]);
  };
};