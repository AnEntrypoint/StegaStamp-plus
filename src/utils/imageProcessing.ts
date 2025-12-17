export async function loadImage(src: string | File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;

    if (typeof src === 'string') {
      img.src = src;
    } else {
      img.src = URL.createObjectURL(src);
    }
  });
}

export function imageToTensor(
  img: HTMLImageElement,
  width: number,
  height: number
): { data: Float32Array; dims: number[] } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = new Float32Array(3 * width * height);

  for (let i = 0; i < width * height; i++) {
    data[i * 3] = imageData.data[i * 4] / 255.0;
    data[i * 3 + 1] = imageData.data[i * 4 + 1] / 255.0;
    data[i * 3 + 2] = imageData.data[i * 4 + 2] / 255.0;
  }

  return {
    data,
    dims: [1, height, width, 3]
  };
}

export function tensorToImage(
  data: Float32Array,
  width: number,
  height: number
): ImageData {
  const imageData = new ImageData(width, height);
  const pixels = imageData.data;

  for (let i = 0; i < width * height; i++) {
    const r = Math.round(Math.max(0, Math.min(255, data[i * 3] * 255)));
    const g = Math.round(Math.max(0, Math.min(255, data[i * 3 + 1] * 255)));
    const b = Math.round(Math.max(0, Math.min(255, data[i * 3 + 2] * 255)));

    pixels[i * 4] = r;
    pixels[i * 4 + 1] = g;
    pixels[i * 4 + 2] = b;
    pixels[i * 4 + 3] = 255;
  }

  return imageData;
}

export function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(blob!);
    }, 'image/png');
  });
}

export function bitsToBytes(bits: boolean[]): Uint8Array {
  const bytes = new Uint8Array(Math.ceil(bits.length / 8));
  for (let i = 0; i < bits.length; i++) {
    if (bits[i]) {
      bytes[Math.floor(i / 8)] |= 1 << (7 - (i % 8));
    }
  }
  return bytes;
}

export function bytesToBits(bytes: Uint8Array): boolean[] {
  const bits: boolean[] = [];
  for (let i = 0; i < bytes.length; i++) {
    for (let j = 7; j >= 0; j--) {
      bits.push((bytes[i] & (1 << j)) !== 0);
    }
  }
  return bits;
}

export function stringToBits(str: string): boolean[] {
  const encoded = new TextEncoder().encode(str);
  return bytesToBits(encoded);
}

export function bitsToString(bits: boolean[]): string {
  const bytes = bitsToBytes(bits);
  return new TextDecoder().decode(bytes);
}

export function padBits(bits: boolean[], targetLen: number): boolean[] {
  const padded = [...bits];
  while (padded.length < targetLen) {
    padded.push(false);
  }
  return padded.slice(0, targetLen);
}
