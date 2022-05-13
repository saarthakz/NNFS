export function sigmoid(inputs: number[]): number[] {
  return inputs.map((input) => 1 / (1 + Math.exp(-input)));
};

export function dSigmoid(outputs: number[]): number[] {
  return outputs.map((output) => output * (1 - output));
};