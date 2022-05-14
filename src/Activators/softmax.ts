export function softmax(inputs: number[]): number[] {
  const expArr = inputs.map((val) => Math.exp(val));
  let denominator = 0;
  expArr.forEach((val) => denominator += val);
  const softmaxArr = expArr.map((val) => val / denominator);
  return softmaxArr;
};


