class ActivationLayer {

  #input: number[] = [];
  #output: number[] = [];
  #func;
  #derivative;

  constructor(func: (input: number[]) => number[], derivative: (input: number[]) => number[]) {
    this.#func = func;
    this.#derivative = derivative;
  };

  forward(input: number[]) {
    this.#input = input;
    return this.#func(input);
  };

  backward(outputGradient: number[], learningRate: number) {
    const inputGradient: number[] = new Array(this.#input.length);
    const derivatives = this.#derivative(this.#input);
    for (let idx = 0; idx < this.#input.length; idx++) {
      inputGradient[idx] = outputGradient[idx] * derivatives[idx];
    };
    return inputGradient;
  };
};