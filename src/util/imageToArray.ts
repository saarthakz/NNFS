import Jimp from "jimp";

export default async function (path: string): Promise<Array<number>> {
  const image = await Jimp.read(path);
  const imageMatrix = new Array<number>(image.bitmap.data.length / 4);

  for (let idx = 0; idx < image.bitmap.data.length; idx = idx + 4) {
    const RGB = image.bitmap.data[idx] / 255;
    imageMatrix[idx / 4] = RGB;
  };

  return imageMatrix;
};