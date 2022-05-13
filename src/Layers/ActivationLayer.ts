export class HiddenActivationLayer {

  #input: number[] = [];
  #output: number[] = [];
  #func;
  #derivative;

  constructor(func: (input: number[]) => number[], derivative: (output: number[]) => number[]) {
    this.#func = func;
    this.#derivative = derivative;
  };

  forward(input: number[]) {
    this.#input = input;
    this.#output = this.#func(input);
    return this.#output;
  };

  backward(outputGradient: number[], learningRate: number) {
    const inputGradient: number[] = new Array(this.#input.length);
    const derivatives = this.#derivative(this.#output);
    for (let idx = 0; idx < this.#input.length; idx++) {
      inputGradient[idx] = outputGradient[idx] * derivatives[idx];
    };
    return inputGradient;
  };
};

export class OutputActivationLayer {

  #input: number[] = [];
  #output: number[] = [];
  #func;
  #derivative;

  constructor(func: (input: number[]) => number[], derivative: (output: number[]) => number[]) {
    this.#func = func;
    this.#derivative = derivative;
  };

  forward(input: number[]) {
    this.#input = input;
    this.#output = this.#func(input);
    return this.#output;
  };

  backward(outputGradient: number[], learningRate: number) {
    const inputGradient: number[] = new Array(this.#input.length);
    const derivatives = this.#derivative(this.#output);
    for (let idx = 0; idx < this.#input.length; idx++) {
      inputGradient[idx] = outputGradient[idx] * derivatives[idx];
    };
    return inputGradient;
  };
};