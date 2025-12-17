import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  encoderPath: string;
  decoderPath: string;
  detectorPath?: string;
  useWebGPU?: boolean;
}

export class StegaStampModel {
  private encoder: ort.InferenceSession | null = null;
  private decoder: ort.InferenceSession | null = null;
  private detector: ort.InferenceSession | null = null;
  private config: ModelConfig;

  constructor(config: ModelConfig) {
    this.config = config;
    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
  }

  private getProviders(): string[] {
    if (this.config.useWebGPU) {
      return ['webgpu', 'wasm'];
    }
    return ['wasm'];
  }

  async init(): Promise<void> {
    try {
      const providers = this.getProviders();
      console.log('Loading encoder model with providers:', providers);
      this.encoder = await ort.InferenceSession.create(
        this.config.encoderPath,
        { executionProviders: providers }
      );

      console.log('Loading decoder model...');
      this.decoder = await ort.InferenceSession.create(
        this.config.decoderPath,
        { executionProviders: providers }
      );

      if (this.config.detectorPath) {
        console.log('Loading detector model...');
        this.detector = await ort.InferenceSession.create(
          this.config.detectorPath,
          { executionProviders: providers }
        );
      }
    } catch (error) {
      console.error('Failed to initialize models:', error);
      throw error;
    }
  }

  async encode(
    image: Float32Array,
    secret: Uint8Array,
    imageDims: number[]
  ): Promise<Float32Array> {
    if (!this.encoder) throw new Error('Encoder not initialized');

    const secretFloat = new Float32Array(secret.length);
    for (let i = 0; i < secret.length; i++) {
      secretFloat[i] = secret[i];
    }

    const feeds: Record<string, ort.Tensor> = {
      'image': new ort.Tensor('float32', image, imageDims),
      'secret': new ort.Tensor('float32', secretFloat, [1, secret.length])
    };

    const results = await this.encoder.run(feeds);
    const output = results[Object.keys(results)[0]] as ort.Tensor;
    return output.data as Float32Array;
  }

  async decode(
    image: Float32Array,
    imageDims: number[]
  ): Promise<{ bits: Uint8Array; confidence: number }> {
    if (!this.decoder) throw new Error('Decoder not initialized');

    const feeds: Record<string, ort.Tensor> = {
      'image': new ort.Tensor('float32', image, imageDims)
    };

    const results = await this.decoder.run(feeds);
    const outputs = Object.values(results) as ort.Tensor[];

    const bitsFloat = outputs[0].data as Float32Array;
    const bits = new Uint8Array(bitsFloat.length);
    for (let i = 0; i < bitsFloat.length; i++) {
      bits[i] = bitsFloat[i] > 0.5 ? 1 : 0;
    }
    const confidence = (outputs[1].data as Float32Array)[0];

    return { bits, confidence };
  }

  async detect(
    image: Float32Array,
    imageDims: number[]
  ): Promise<{
    mask: Float32Array;
    bbox: [number, number, number, number];
  }> {
    if (!this.detector) throw new Error('Detector not initialized');

    const feeds: Record<string, ort.Tensor> = {
      'image': new ort.Tensor('float32', image, imageDims)
    };

    const results = await this.detector.run(feeds);
    const outputs = Object.values(results) as ort.Tensor[];

    const mask = outputs[0].data as Float32Array;
    const bbox = Array.from(outputs[1].data as Float32Array) as [number, number, number, number];

    return { mask, bbox };
  }

  dispose(): void {
    this.encoder?.release();
    this.decoder?.release();
    this.detector?.release();
  }
}

export default StegaStampModel;
