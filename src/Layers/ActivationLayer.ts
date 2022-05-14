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

  backward(outputGradient: number[]) {
    const inputGradient: number[] = new Array(this.#input.length);
    const derivatives = this.#derivative(this.#output);
    for (let idx = 0; idx < this.#input.length; idx++) {
      inputGradient[idx] = outputGradient[idx] * derivatives[idx];
    };
    return inputGradient;
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