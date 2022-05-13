export function reLU(inputs: number[]): number[] {
  return inputs.map((val) => Math.max(val, 0));
};

export function dReLU(outputs: number[]): number[] {
  return outputs.map((output) => {
    if (output > 0) return 1;
    return 0;
  });
};
